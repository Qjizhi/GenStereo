import argparse
import logging
import math
import os
import os.path as osp
import random
import warnings
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import diffusers
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm
from diffusers.image_processor import VaeImageProcessor
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from transformers import CLIPVisionModelWithProjection

from genstereo.dataset.dataset_genstereo import StereoGenDataset
from genstereo.models.mutual_self_attention import ReferenceAttentionControl
from genstereo import AdaptiveFusionLayer
from genstereo.models.pose_guider import PoseGuider
from genstereo.models.unet_2d_condition import UNet2DConditionModel
from genstereo.models.unet_3d import UNet3DConditionModel
from genstereo.utils.util import delete_additional_ckpt, seed_everything

warnings.filterwarnings("ignore")

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
        timesteps
    ].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
        device=timesteps.device
    )[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


def main(cfg):
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
        mixed_precision=cfg.solver.mixed_precision,
        log_with="mlflow",
        project_dir="./mlruns",
        kwargs_handlers=[kwargs],
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        seed_everything(cfg.seed)

    exp_name = cfg.exp_name
    save_dir = f"{cfg.output_dir}/{exp_name}"
    if accelerator.is_main_process and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif cfg.weight_dtype == "fp32":
        weight_dtype = torch.float32
    else:
        raise ValueError(
            f"Do not support weight dtype: {cfg.weight_dtype} during training"
        )

    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    val_noise_scheduler = DDIMScheduler(**sched_kwargs)
    sched_kwargs.update({"beta_schedule": "scaled_linear"})
    train_noise_scheduler = DDIMScheduler(**sched_kwargs)

    ############################ 0. VAE ################################
    vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to(
        dtype=weight_dtype, device="cuda"
    )
    vae_scale_factor = \
        2 ** (len(vae.config.block_out_channels) - 1)

    vae_image_processor = VaeImageProcessor(
        vae_scale_factor=vae_scale_factor, do_convert_rgb=True, do_normalize=False
    )  # to [-1, 1]
    from jaxtyping import Float
    from torch import Tensor
    def decode_latents(latents: Float[Tensor, 'B C H W']) -> Float[Tensor, 'B C H W']:
        latents = 1 / 0.18215 * latents
        rgb = []
        for frame_idx in range(latents.shape[0]):
            rgb.append(vae.decode(latents[frame_idx : frame_idx + 1]).sample)
        rgb = torch.cat(rgb)
        rgb = (rgb / 2 + 0.5).clamp(0, 1)
        return rgb.squeeze(2)

    ############################ 1. CLIP ################################

    clip_image_processor = CLIPImageProcessor()

    # 2. Image encoder.
    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        cfg.image_encoder_path,
    ).to(dtype=weight_dtype, device="cuda")

    ############################ 2. UNet ################################

    # Reference Unet.
    reference_unet = UNet2DConditionModel.from_pretrained(
        cfg.base_model_path,
        subfolder="unet",
    ).to(device="cuda")

    # Denoising Unet.
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        osp.join(cfg.base_model_path, 'unet/config.json'),
        osp.join(cfg.base_model_path, 'unet/diffusion_pytorch_model.bin')
    ).to(device="cuda")

    ############################ 3. Pose Guider ################################
    if cfg.use_coords and cfg.use_wapred:
        pose_guider = PoseGuider(
            conditioning_embedding_channels=320,
            conditioning_channels=14,
        ).to(device="cuda")
    elif cfg.use_coords and not cfg.use_wapred:
        pose_guider = PoseGuider(
            conditioning_embedding_channels=320,
            conditioning_channels=11,
        ).to(device="cuda")
    else:
        pose_guider = PoseGuider(
            conditioning_embedding_channels=320,
            conditioning_channels=4,
        ).to(device="cuda")

    # Freeze
    vae.requires_grad_(False)
    image_enc.requires_grad_(False)

    # Explictly declare training models
    denoising_unet.requires_grad_(True)
    #  Some top layer parames of reference_unet don't need grad
    for name, param in reference_unet.named_parameters():
        if "up_blocks.3" in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)
    # reference_unet.requires_grad_(True)
    pose_guider.requires_grad_(True)

    reference_control_writer = ReferenceAttentionControl(
            reference_unet,
            do_classifier_free_guidance=False,
            mode='write',
            batch_size=cfg.data.train_bs,
            fusion_blocks='full',
            feature_fusion_type='attention_full_sharing'
    )
    reference_control_reader = ReferenceAttentionControl(
            denoising_unet,
            do_classifier_free_guidance=False,
            mode='read',
            batch_size=cfg.data.train_bs,
            fusion_blocks='full',
            feature_fusion_type='attention_full_sharing'
    )
    fusion_layer = AdaptiveFusionLayer()

    class Net(nn.Module):
        def __init__(
            self,
            reference_unet: UNet2DConditionModel,
            denoising_unet: UNet3DConditionModel,
            pose_guider: PoseGuider,
            reference_control_writer,
            reference_control_reader,
            fusion_layer: AdaptiveFusionLayer,
        ):
            super().__init__()
            self.reference_unet = reference_unet
            self.denoising_unet = denoising_unet
            self.pose_guider = pose_guider
            self.reference_control_writer = reference_control_writer
            self.reference_control_reader = reference_control_reader
            self.fusion_layer = fusion_layer

        def forward(
            self,
            noisy_latents,
            timesteps,
            ref_image_latents,
            clip_image_embeds,
            pose_fea,
            pose_fea_2,
            correspondence,
            cfg,
            batch,
            uncond_fwd: bool = False,
        ):
            # pose_cond_tensor = pose_fea.to(device="cuda")
            # pose_fea = self.pose_guider(pose_cond_tensor)

            if not uncond_fwd:
                ref_timesteps = torch.zeros_like(timesteps)
                # bs = ref_image_latents.shape[0]
                self.reference_unet(
                    ref_image_latents,
                    ref_timesteps,
                    encoder_hidden_states=clip_image_embeds,
                    pose_cond_fea=pose_fea,
                    return_dict=False,
                )
                self.reference_control_reader.update(self.reference_control_writer, correspondence)

            noise_pred = self.denoising_unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=clip_image_embeds,
                pose_cond_fea=pose_fea_2,
                return_dict=False,
            )[0]
            decoded_pred = decode_latents(noise_pred.squeeze(2))
            if cfg.add_fusion:
                mask = batch["mask"].to("cuda").to(dtype=weight_dtype)
                warped_image = batch["converted_right"].to(dtype=weight_dtype).to("cuda")*0.5+0.5
                fusion_pred = fusion_layer(decoded_pred, warped_image, mask)
                return noise_pred, fusion_pred

            return noise_pred, decoded_pred


    net = Net(
        reference_unet,
        denoising_unet,
        pose_guider,
        reference_control_writer,
        reference_control_reader,
        fusion_layer,
    )

    if cfg.solver.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            reference_unet.enable_xformers_memory_efficient_attention()
            denoising_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if cfg.solver.gradient_checkpointing:
        reference_unet.enable_gradient_checkpointing()
        denoising_unet.enable_gradient_checkpointing()

    if cfg.solver.scale_lr:
        learning_rate = (
            cfg.solver.learning_rate
            * cfg.solver.gradient_accumulation_steps
            * cfg.data.train_bs
            * accelerator.num_processes
        )
    else:
        learning_rate = cfg.solver.learning_rate

    # Initialize the optimizer
    if cfg.solver.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    trainable_params = list(filter(lambda p: p.requires_grad, net.parameters()))
    optimizer = optimizer_cls(
        trainable_params,
        lr=learning_rate,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,
    )

    # Scheduler
    lr_scheduler = get_scheduler(
        cfg.solver.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.solver.lr_warmup_steps
        * cfg.solver.gradient_accumulation_steps,
        num_training_steps=cfg.solver.max_train_steps
        * cfg.solver.gradient_accumulation_steps,
    )
    train_dataset = StereoGenDataset(json_files=cfg.data.meta_paths, img_size=cfg.data.img_size, drop_ratio=cfg.data.drop_ratio, debug=cfg.debug, use_coords=cfg.use_coords, use_wapred=cfg.use_wapred)
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.data.train_bs, shuffle=True, num_workers=16
    )

    # Prepare everything with our `accelerator`.
    (
        net,
        optimizer,
        train_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        net,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.solver.gradient_accumulation_steps
    )
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(
        cfg.solver.max_train_steps / num_update_steps_per_epoch
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run_time = datetime.now().strftime("%Y%m%d-%H%M")
        accelerator.init_trackers(
            cfg.exp_name,
            init_kwargs={"mlflow": {"run_name": run_time}},
        )
        # dump config file
        # mlflow.log_dict(OmegaConf.to_container(cfg), "config.yaml")

    # Train!
    total_batch_size = (
        cfg.data.train_bs
        * accelerator.num_processes
        * cfg.solver.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.data.train_bs}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {cfg.solver.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {cfg.solver.max_train_steps}")
    logger.info(f" Pixel loss: {cfg.solver.pixel_loss}")
    logger.info(f" drop_ratio: {cfg.data.drop_ratio}")
    logger.info(f" use_coords: {cfg.use_coords}")
    logger.info(f" use_wapred: {cfg.use_wapred}")
    logger.info(f" add_fusion: {cfg.add_fusion}")

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint != "latest":
            resume_dir = cfg.resume_from_checkpoint
        else:
            resume_dir = save_dir
        # Get the most recent checkpoint
        dirs = os.listdir(resume_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1]
        accelerator.load_state(os.path.join(resume_dir, path))
        accelerator.print(f"Resuming from checkpoint {path}")
        global_step = int(path.split("-")[1])

        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = global_step % num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, cfg.solver.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(net):
                # Convert videos to latent space
                pixel_values = batch["target"].to(dtype=weight_dtype)
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents.unsqueeze(2)  # (b, c, 1, h, w)
                    latents = latents * 0.18215

                noise = torch.randn_like(latents)
                if cfg.noise_offset > 0.0:
                    noise += cfg.noise_offset * torch.randn(
                        (noise.shape[0], noise.shape[1], 1, 1, 1),
                        device=noise.device,
                    )

                bsz = latents.shape[0]
                # Sample a random timestep
                timesteps = torch.randint(
                    0,
                    train_noise_scheduler.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                # add noise
                noisy_latents = train_noise_scheduler.add_noise(
                    latents, noise, timesteps
                )

                # Get ref_image_latents and image_prompt_embeds
                ref_img = batch["source"].to(dtype=weight_dtype)

                with torch.no_grad():
                    ref_image_latents = vae.encode(ref_img).latent_dist.sample()  # (bs, d, 64, 64), [-1, 1], torch.Size([4, 4, 64, 64])
                    ref_image_latents = ref_image_latents * 0.18215

                    clip_image_embeds = image_enc(
                        batch["clip_images"].to("cuda", dtype=weight_dtype)
                    ).image_embeds
                    image_prompt_embeds = clip_image_embeds.unsqueeze(1)  # (bs, 1, d)
                    image_prompt_embeds = F.pad(image_prompt_embeds, (0, 256), "constant", 0)  # Now shape is (bs, 1, 1024)


                uncond_fwd = random.random() < cfg.uncond_ratio

                image_prompt_embeds_list = []
                for batch_idx, image_prompt_embed in enumerate(image_prompt_embeds):
                    if uncond_fwd:
                        image_prompt_embeds_list.append(torch.zeros_like(image_prompt_embed))
                    else:
                        image_prompt_embeds_list.append(image_prompt_embed)
                image_prompt_embeds = torch.stack(image_prompt_embeds_list, dim=0).to(
                    dtype=image_enc.dtype, device=image_enc.device
                )
                # Get pose_fea, pose_fea_2
                src_embed = batch["src_coords_embed"].to("cuda")
                trg_embed = batch["trg_coords_embed"].to("cuda")
                pose_cond_tensor = src_embed.unsqueeze(2)
                pose_cond_tensor = pose_cond_tensor
                pose_cond_tensor_2 = trg_embed.unsqueeze(2)
                pose_cond_tensor_2 = pose_cond_tensor_2
                pose_fea = pose_guider(pose_cond_tensor)
                pose_fea_2 = pose_guider(pose_cond_tensor_2)
                
                pose_fea = pose_fea[:, :, 0, ...]

                # Get the target for loss depending on the prediction type
                if train_noise_scheduler.prediction_type == "epsilon":
                    target = noise
                elif train_noise_scheduler.prediction_type == "v_prediction":
                    target = train_noise_scheduler.get_velocity(
                        latents, noise, timesteps
                    )
                else:
                    raise ValueError(
                        f"Unknown prediction type {train_noise_scheduler.prediction_type}"
                    )

                # Get correspondence
                correspondence = batch["correspondence"].to("cuda")
                # import IPython; IPython.embed()
                model_pred, final_pred = net(
                    noisy_latents,
                    timesteps,
                    ref_image_latents,
                    image_prompt_embeds,
                    pose_fea,
                    pose_fea_2,
                    correspondence,
                    cfg,
                    batch,
                    uncond_fwd,
                )

                if cfg.snr_gamma == 0:
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="mean"
                    )
                else:
                    snr = compute_snr(train_noise_scheduler, timesteps)
                    if train_noise_scheduler.config.prediction_type == "v_prediction":
                        # Velocity objective requires that we add one to SNR values before we divide by them.
                        snr = snr + 1
                    mse_loss_weights = (
                        torch.stack(
                            [snr, cfg.snr_gamma * torch.ones_like(timesteps)], dim=1
                        ).min(dim=1)[0]
                        / snr
                    )
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="none"
                    )
                    loss = (
                        loss.mean(dim=list(range(1, len(loss.shape))))
                        * mse_loss_weights
                    )
                    loss = loss.mean()

                    if cfg.solver.pixel_loss:
                        decoded_target = decode_latents(target.squeeze(2))
                        pixel_loss = F.mse_loss(final_pred.float(), decoded_target.float(), reduction="none")  # `target_image` should be in the same pixel space
                        pixel_loss = pixel_loss.mean(dim=list(range(1, len(pixel_loss.shape))))  # Averaging over spatial dimensions
                        pixel_loss = pixel_loss.mean()  # Final mean for the pixel-level loss
                        combined_loss = loss + pixel_loss
                        loss = combined_loss


                # loss.backward(retain_graph=True)  # Retain graph for subsequent backward calls

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(cfg.data.train_bs)).mean()
                train_loss += avg_loss.item() / cfg.solver.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        trainable_params,
                        cfg.solver.max_grad_norm,
                    )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                reference_control_reader.clear()
                reference_control_writer.clear()
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
                if global_step % cfg.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(save_dir, f"checkpoint-{global_step}")
                        delete_additional_ckpt(save_dir, 1)
                        accelerator.save_state(save_path)

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.solver.max_train_steps:
                break

        # save model after each epoch
        if (
            epoch + 1
        ) - cfg.save_model_epoch_interval >= -1 and accelerator.is_main_process:
            unwrap_net = accelerator.unwrap_model(net)
            save_checkpoint(
                unwrap_net.reference_unet,
                save_dir,
                "reference_unet",
                global_step,
                total_limit=23,
            )
            save_checkpoint(
                unwrap_net.denoising_unet,
                save_dir,
                "denoising_unet",
                global_step,
                total_limit=23,
            )
            save_checkpoint(
                unwrap_net.pose_guider,
                save_dir,
                "pose_guider",
                global_step,
                total_limit=23,
            )
            save_checkpoint(
                fusion_layer,
                save_dir,
                "fusion_layer",
                global_step,
                total_limit=23,
            )            

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


def save_checkpoint(model, save_dir, prefix, ckpt_num, total_limit=None):
    save_path = osp.join(save_dir, f"{prefix}-{ckpt_num}.pth")

    if total_limit is not None:
        checkpoints = os.listdir(save_dir)
        checkpoints = [d for d in checkpoints if d.startswith(prefix)]
        checkpoints = sorted(
            checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0])
        )

        if len(checkpoints) >= total_limit:
            num_to_remove = len(checkpoints) - total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]
            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(save_dir, removing_checkpoint)
                os.remove(removing_checkpoint)

    state_dict = model.state_dict()
    torch.save(state_dict, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train.yaml")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    main(config)