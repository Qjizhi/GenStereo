from os.path import join
from typing import Union, Optional, List, Dict, Tuple, Any
from dataclasses import dataclass
import inspect

from omegaconf import OmegaConf, DictConfig
from jaxtyping import Float
from PIL import Image

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from einops import rearrange, repeat

from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.image_processor import VaeImageProcessor
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

from .models import (
    PoseGuider,
    UNet2DConditionModel,
    UNet3DConditionModel,
    ReferenceAttentionControl
)
from .ops import convert_left_to_right, convert_left_to_right_torch

class AdaptiveFusionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(7, 1, kernel_size=3, stride=1, padding=1),  # (B, 7, H, W) → (B, 1, H, W)
            nn.Sigmoid()  # Ensures weights are between 0 and 1
        )

    def forward(self, generated_image, warped_image, mask):
        # Compute fusion weights
        fusion_input = torch.cat([generated_image, warped_image, mask], dim=1)  # (B, 7, H, W)
        weights = self.fusion_layer(fusion_input)  # (B, 1, H, W)
        fused_output = mask * weights * warped_image + (1 - mask * weights) * generated_image
        return fused_output 

class GenStereo():
    @dataclass
    class Config():
        pretrained_model_path: str = ''
        checkpoint_name: str = ''
        half_precision_weights: bool = False
        height: int = 768
        width: int = 768
        num_inference_steps: int = 50
        guidance_scale: float = 1.5
    cfg: Config

    class Embedder():
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.create_embedding_fn()

        def create_embedding_fn(self) -> None:
            embed_fns = []
            d = self.kwargs['input_dims']
            out_dim = 0
            if self.kwargs['include_input']:
                embed_fns.append(lambda x : x)
                out_dim += d

            max_freq = self.kwargs['max_freq_log2']
            N_freqs = self.kwargs['num_freqs']

            if self.kwargs['log_sampling']:
                freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
            else:
                freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

            for freq in freq_bands:
                for p_fn in self.kwargs['periodic_fns']:
                    embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                    out_dim += d

            self.embed_fns = embed_fns
            self.out_dim = out_dim

        def embed(self, inputs) -> Tensor:
            return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

    def __init__(
        self,
        cfg: Optional[Union[dict, DictConfig]] = None,
        device: Optional[str] = 'cuda:0',
        sd_version: Optional[str] = 'v2.1'
    ) -> None:
        self.cfg = OmegaConf.structured(self.Config(**cfg))
        self.model_path = join(
            self.cfg.pretrained_model_path, self.cfg.checkpoint_name
        )
        self.device = device
        self.sd_version = sd_version
        self.configure()
        self.transform_pixels = transforms.Compose([
            transforms.ToTensor(),  # Converts image to Tensor
            transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
        ])
        if self.sd_version == "v1.5":
            self.cfg.height = 512
            self.cfg.width = 512
        elif self.sd_version == "v2.1":
            self.cfg.height = 768
            self.cfg.width = 768
        else:
            raise ValueError(f"Unknown SD version: {self.sd_version}")

    def configure(self) -> None:
        print(f"Loading GenStereo...")

        # Configurations.
        self.dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        # Load models.
        self.load_models()

        # Timestep
        self.scheduler.set_timesteps(
            self.cfg.num_inference_steps, device=self.device)
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps

        print(f"Loaded GenStereo.")

    def load_models(self) -> None:
        # VAE.
        self.vae = AutoencoderKL.from_pretrained(
            join(self.cfg.pretrained_model_path, 'sd-vae-ft-mse')
        ).to(self.device, dtype=self.dtype)

        # Image processor.
        self.vae_scale_factor = \
            2 ** (len(self.vae.config.block_out_channels) - 1)
        self.vae_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True
        )
        self.clip_image_processor = CLIPImageProcessor()

        # Image encoder.
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            join(self.cfg.pretrained_model_path, 'image_encoder')
        ).to(self.device, dtype=self.dtype)

        # Reference Unet.
        self.reference_unet = UNet2DConditionModel.from_config(
            UNet2DConditionModel.load_config(
                join(self.model_path, 'config.json')
        )).to(self.device, dtype=self.dtype)
        self.reference_unet.load_state_dict(torch.load(
            join(self.model_path, 'reference_unet.pth'),
            map_location= 'cpu'),
        )

        # Denoising Unet.
        self.denoising_unet = UNet3DConditionModel.from_pretrained_2d(
            join(self.model_path, 'config.json'),
            join(self.model_path, 'denoising_unet.pth')
        ).to(self.device, dtype=self.dtype)
        self.unet_in_channels = self.denoising_unet.config.in_channels

        # Pose guider.
        self.pose_guider = PoseGuider(
            conditioning_embedding_channels=320,
            # TODO
            conditioning_channels=14,
        ).to(self.device, dtype=self.dtype)
        self.pose_guider.load_state_dict(torch.load(
            join(self.model_path, 'pose_guider.pth'),
            map_location='cpu'),
        )

        # Noise scheduler
        sched_kwargs = OmegaConf.to_container(OmegaConf.create({
            'num_train_timesteps': 1000,
            'beta_start': 0.00085,
            'beta_end': 0.012,
            'beta_schedule': 'scaled_linear',
            'steps_offset': 1,
            'clip_sample': False
        }))
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing='trailing',
            prediction_type='v_prediction',
        )
        self.scheduler = DDIMScheduler(**sched_kwargs)

        self.vae.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.reference_unet.requires_grad_(False)
        self.denoising_unet.requires_grad_(False)
        self.pose_guider.requires_grad_(False)

        # Coordinates embedding.
        self.embedder = self.get_embedder(2)

    def get_embedder(self, multires):
        embed_kwargs = {
            'include_input' : True,
            'input_dims' : 2,
            'max_freq_log2' : multires-1,
            'num_freqs' : multires,
            'log_sampling' : True,
            'periodic_fns' : [torch.sin, torch.cos],
        }

        embedder_obj = self.Embedder(**embed_kwargs)
        embed = lambda x, eo=embedder_obj : eo.embed(x)
        return embed

    def __call__(
        self,
        src_image: Image,
        src_disparity: Float[Tensor, 'B C H W'],
        ratio
    ) -> Dict[str, Tensor]:
        """ Perform NVS.
        """

        src_image_pil = src_image
        src_image = self.transform_pixels(src_image).unsqueeze(0).to(self.device, dtype=self.dtype)
        batch_size = src_image.shape[0]
        # Rearrange and resize.
        src_image = self.preprocess_image(src_image)
        src_disparity = self.preprocess_image(src_disparity)

        pipe_args = dict(
            src_image=src_image,
            src_image_pil=src_image_pil,
            src_disparity=src_disparity,
            ratio=ratio
        )

        # Prepare inputs.
        conditions, renders = self.prepare_conditions_stereogen(**pipe_args)

        # NVS.
        latents_clean = self.perform_nvs(
            **pipe_args,
            **conditions,
            **renders
        )

        # Decode to images.
        synthesized = self.decode_latents(latents_clean)

        inference_out = {
            'synthesized': synthesized,
            'warped': renders['warped'],
            'mask': renders['mask'],
            'correspondence': conditions['correspondence']
        }

        return inference_out

    def preprocess_image(
        self,
        image: Float[Tensor, 'B C H W']
    ) -> Float[Tensor, 'B C H W']:
        image = F.interpolate(
            image, (self.cfg.height, self.cfg.width)
        )
        return image

    def get_image_prompt(
        self,
        src_image_pil: Image
    ) -> Float[Tensor, '2 B L']:
        clip_image = self.clip_image_processor(
            images=src_image_pil, return_tensors="pt"
        ).pixel_values

        clip_image_embeds = self.image_encoder(
            clip_image.to(self.device, dtype=self.image_encoder.dtype)
        ).image_embeds

        image_prompt_embeds = clip_image_embeds.unsqueeze(1)
        if self.sd_version == "v2.1":
            image_prompt_embeds = F.pad(image_prompt_embeds, (0, 256), "constant", 0)  # Now shape is (bs, 1, 1024)
        uncond_image_prompt_embeds = torch.zeros_like(image_prompt_embeds)

        image_prompt_embeds = torch.cat(
            [uncond_image_prompt_embeds, image_prompt_embeds], dim=0
        )

        return image_prompt_embeds

    def encode_images(
        self,
        rgb: Float[Tensor, 'B C H W']
    ) -> Float[Tensor, 'B C H W']:
        latents = self.vae.encode(rgb).latent_dist.mean  # rgb [-1, 1]
        latents = latents * 0.18215
        return latents

    def decode_latents(
        self,
        latents: Float[Tensor, 'B C H W']
    ) -> Float[Tensor, 'B C H W']:
        latents = 1 / 0.18215 * latents
        rgb = []
        for frame_idx in range(latents.shape[0]):
            rgb.append(self.vae.decode(latents[frame_idx : frame_idx + 1]).sample)
        rgb = torch.cat(rgb)
        rgb = (rgb / 2 + 0.5).clamp(0, 1)
        return rgb.squeeze(2)

    def get_reference_controls(
        self,
        batch_size: int
    ) -> Tuple[ReferenceAttentionControl, ReferenceAttentionControl]:
        reader = ReferenceAttentionControl(
            self.denoising_unet,
            do_classifier_free_guidance=True,
            mode='read',
            batch_size=batch_size,
            fusion_blocks='full',
            feature_fusion_type='attention_full_sharing'
        )
        writer = ReferenceAttentionControl(
            self.reference_unet,
            do_classifier_free_guidance=True,
            mode='write',
            batch_size=batch_size,
            fusion_blocks='full',
            feature_fusion_type='attention_full_sharing'
        )

        return reader, writer

    def prepare_extra_step_kwargs(
        self,
        generator,
        eta
    ) -> Dict[str, Any]:
        accepts_eta = 'eta' in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs['eta'] = eta

        # check if the scheduler accepts generator
        accepts_generator = 'generator' in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs['generator'] = generator
        return extra_step_kwargs

    def get_pose_features(
        self,
        src_embed: Float[Tensor, 'B C H W'],
        trg_embed: Float[Tensor, 'B C H W'],
        do_classifier_guidance: bool = True
    ) -> Tuple[Tensor, Tensor]:
        pose_cond_tensor = src_embed.unsqueeze(2)
        pose_cond_tensor = pose_cond_tensor.to(
            device=self.device, dtype=self.pose_guider.dtype
        )
        pose_cond_tensor_2 = trg_embed.unsqueeze(2)
        pose_cond_tensor_2 = pose_cond_tensor_2.to(
            device=self.device, dtype=self.pose_guider.dtype
        )
        pose_fea = self.pose_guider(pose_cond_tensor)
        pose_fea_2 = self.pose_guider(pose_cond_tensor_2)

        if do_classifier_guidance:
            pose_fea = torch.cat([pose_fea] * 2)
            pose_fea_2 = torch.cat([pose_fea_2] * 2)

        return pose_fea, pose_fea_2

    @torch.no_grad()
    def prepare_conditions_stereogen(
        self,
        src_image: Float[Tensor, 'B C H W'],
        src_image_pil,
        src_disparity: Float[Tensor, 'B C H W'],
        ratio
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        # Prepare inputs.
        B = src_image.shape[0]
        H, W = src_image.shape[2:4]

        # Coordinate grids.
        grid: Float[Tensor, 'H W C'] = torch.stack(torch.meshgrid(
            torch.arange(W), torch.arange(H), indexing='xy'), dim=-1
        ).to(self.device, dtype=self.dtype)

        # Coordinates embedding.
        coords = torch.stack((grid[..., 0]/H, grid[..., 1]/W), dim=-1)
        embed = repeat(self.embedder(coords), 'h w c -> b c h w', b=B)

        warped_embed, mask, warped_image, disparity = convert_left_to_right_torch(embed.squeeze(0), src_disparity.squeeze(0), src_image.squeeze(0), ratio)
        warped_embed, mask, warped_image = warped_embed.unsqueeze(0), mask.unsqueeze(0).unsqueeze(0), warped_image.unsqueeze(0)
        # mask == 1 where there's no pixel

        # Conditions 14
        src_coords_embed = torch.cat(
            [embed, torch.zeros_like(mask, device=mask.device), src_image], dim=1)
        trg_coords_embed = torch.cat([warped_embed, mask, warped_image], dim=1)

        # Conditions 11
        # src_coords_embed = torch.cat(
        #     [embed, torch.zeros_like(mask, device=mask.device)], dim=1)
        # trg_coords_embed = torch.cat([warped_embed, mask], dim=1)

        # # Conditions 4
        # src_coords_embed = torch.cat(
        #     [torch.zeros_like(mask, device=mask.device), src_image], dim=1)
        # trg_coords_embed = torch.cat([mask, warped_image], dim=1)

        # Outputs.
        conditions = dict(
            src_coords_embed=src_coords_embed,
            trg_coords_embed=trg_coords_embed,
            correspondence=disparity
        )
        renders = dict(
            warped=warped_image,
            mask=1-mask  # mask == 1 where there's a pixel
        )

        return conditions, renders

    def perform_nvs(
        self,
        src_image,
        src_image_pil,
        src_coords_embed,
        trg_coords_embed,
        correspondence,
        warped,
        mask,
        eta: float=0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]]=None,
        **kwargs,
    ) -> Float[Tensor, 'B C H W']:
        batch_size = src_image.shape[0]

        # For the cross attention.
        reference_control_reader, reference_control_writer = \
            self.get_reference_controls(batch_size)

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(
            generator, eta
        )

        with torch.no_grad():
            # Create fake inputs. It'll be replaced by pure noise.
            latents = torch.randn(
                batch_size,
                self.unet_in_channels,
                self.cfg.height // self.vae_scale_factor,
                self.cfg.width // self.vae_scale_factor
            ).to(self.device, dtype=src_image.dtype)
            initial_t = torch.tensor(
                [self.num_train_timesteps - 1] * batch_size
            ).to(self.device, dtype=torch.long)

            # Add noise.
            noise = torch.randn_like(latents)
            latents_noisy_start = self.scheduler.add_noise(
                latents, noise, initial_t
            )
            latents_noisy_start = latents_noisy_start.unsqueeze(2)
            # Prepare clip image embeds.
            image_prompt_embeds = self.get_image_prompt(src_image_pil)  # torch.Size([2, 1, 768])

            # Prepare ref image latents.
            ref_image_latents = self.encode_images(src_image)

            # Prepare pose condition image.
            pose_fea, pose_fea_2 = self.get_pose_features(
                src_coords_embed, trg_coords_embed
            )

            pose_fea = pose_fea[:, :, 0, ...]
            # Forward reference images.
            self.reference_unet(
                ref_image_latents.repeat(2, 1, 1, 1),  # torch.Size([bs*2, 4, 64, 64])
                torch.zeros(batch_size * 2).to(ref_image_latents),  # torch.Size([bs*2])
                encoder_hidden_states=image_prompt_embeds,  # torch.Size([bs*2, 1, 768])
                pose_cond_fea=pose_fea,  # torch.Size([bs*2, 320, 64, 64])
                return_dict=False,
            )
            # Update the denosing net with reference features.
            reference_control_reader.update(
                reference_control_writer,
                correspondence=correspondence  # torch.Size([bs, 1, 512, 512])
            )

            timesteps = self.scheduler.timesteps
            latents_noisy = latents_noisy_start
            for t in timesteps:
                # Prepare latents.
                latent_model_input = torch.cat([latents_noisy] * 2)
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                # Denoise.
                noise_pred = self.denoising_unet(
                    latent_model_input,  # torch.Size([bs*2, 4, 1, 64, 64])
                    t,  # torch.Size([])
                    encoder_hidden_states=image_prompt_embeds,  # torch.Size([bs*2, 1, 768])
                    pose_cond_fea=pose_fea_2,  # torch.Size([bs*2, 320, 1, 64, 64])
                    return_dict=False,
                )[0]  # torch.Size([bs*2, 4, 1, 64, 64])

                # CFG.
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

                # t -> t-1
                latents_noisy = self.scheduler.step(
                    noise_pred, t, latents_noisy, **extra_step_kwargs,
                    return_dict=False
                )[0]

            # Noise disappears eventually
            latents_clean = latents_noisy

        reference_control_reader.clear()
        reference_control_writer.clear()

        return latents_clean.squeeze(2)