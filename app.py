import os
from os.path import basename, splitext, join
import tempfile
import gradio as gr
import numpy as np
from PIL import Image
import torch
import cv2
from torchvision.transforms.functional import to_tensor, to_pil_image
from torch import Tensor
from genstereo import GenStereo, AdaptiveFusionLayer
import ssl
from huggingface_hub import hf_hub_download

from extern.DAM2.depth_anything_v2.dpt import DepthAnythingV2
ssl._create_default_https_context = ssl._create_unverified_context

IMAGE_SIZE = 512
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
CHECKPOINT_NAME = 'genstereo'

def download_models():
    models = [
        {
            'repo': 'stabilityai/sd-vae-ft-mse',
            'sub': None,
            'dst': 'checkpoints/sd-vae-ft-mse',
            'files': ['config.json', 'diffusion_pytorch_model.safetensors'],
            'token': None
        },
        {
            'repo': 'lambdalabs/sd-image-variations-diffusers',
            'sub': 'image_encoder',
            'dst': 'checkpoints',
            'files': ['config.json', 'pytorch_model.bin'],
            'token': None
        },
        {
            'repo': 'FQiao/GenStereo',
            'sub': None,
            'dst': 'checkpoints/genstereo',
            'files': ['config.json', 'denoising_unet.pth', 'fusion_layer.pth', 'pose_guider.pth', 'reference_unet.pth'],
            'token': None
        },
        {
            'repo': 'depth-anything/Depth-Anything-V2-Large',
            'sub': None,
            'dst': 'checkpoints',
            'files': [f'depth_anything_v2_vitl.pth'],
            'token': None
        }
    ]

    for model in models:
        for file in model['files']:
            hf_hub_download(
                repo_id=model['repo'],
                subfolder=model['sub'],
                filename=file,
                local_dir=model['dst'],
                token=model['token']
            )

# Setup.
download_models()

# DepthAnythingV2
def get_dam2_model():
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    encoder = 'vitl'
    encoder_size_map = {'vits': 'Small', 'vitb': 'Base', 'vitl': 'Large'}

    if encoder not in encoder_size_map:
        raise ValueError(f"Unsupported encoder: {encoder}. Supported: {list(encoder_size_map.keys())}")

    dam2 = DepthAnythingV2(**model_configs[encoder])
    dam2_checkpoint = f'checkpoints/depth_anything_v2_{encoder}.pth'
    dam2.load_state_dict(torch.load(dam2_checkpoint, map_location='cpu'))
    dam2 = dam2.to(DEVICE).eval()
    return dam2

# GenStereo
def get_genstereo_model():
    genstereo_cfg = dict(
        pretrained_model_path='checkpoints',
        checkpoint_name=CHECKPOINT_NAME,
        half_precision_weights=True
    )
    genstereo = GenStereo(cfg=genstereo_cfg, device=DEVICE)
    return genstereo

# Adaptive Fusion
def get_fusion_model():
    fusion_model = AdaptiveFusionLayer()
    fusion_checkpoint = join('checkpoints', CHECKPOINT_NAME, 'fusion_layer.pth')
    fusion_model.load_state_dict(torch.load(fusion_checkpoint, map_location='cpu'))
    fusion_model = fusion_model.to(DEVICE).eval()
    return fusion_model

# Crop the image to the shorter side.
def crop(img: Image) -> Image:
    W, H = img.size
    if W < H:
        left, right = 0, W
        top, bottom = np.ceil((H - W) / 2.), np.floor((H - W) / 2.) + W
    else:
        left, right = np.ceil((W - H) / 2.), np.floor((W - H) / 2.) + H
        top, bottom = 0, H
    return img.crop((left, top, right, bottom))

def normalize_disp(disp):
    return (disp - disp.min()) / (disp.max() - disp.min())

# Gradio app
with tempfile.TemporaryDirectory() as tmpdir:
    with gr.Blocks(
        title='StereoGen Demo',
        css='img {display: inline;}'
    ) as demo:
        # Internal states.
        src_image = gr.State()
        src_depth = gr.State()

        # Callbacks
        def cb_mde(image_file: str):
            if not image_file:
                # Return None if no image is provided (e.g., when file is cleared).
                return None, None, None, None

            image = crop(Image.open(image_file).convert('RGB'))  # Load image using PIL
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

            image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            dam2 = get_dam2_model()
            depth_dam2 = dam2.infer_image(image_bgr)
            depth = torch.tensor(depth_dam2).unsqueeze(0).unsqueeze(0).float()

            depth_image = cv2.applyColorMap((normalize_disp(depth_dam2) * 255).astype(np.uint8), cv2.COLORMAP_JET)

            return image, depth_image, image, depth

        def cb_generate(image, depth: Tensor, scale_factor):
            norm_disp = normalize_disp(depth.cuda())
            disp = norm_disp * scale_factor / 100 * IMAGE_SIZE

            genstereo = get_genstereo_model()
            fusion_model = get_fusion_model()

            renders = genstereo(
                src_image=image,
                src_disparity=disp,
                ratio=None,
            )
            warped = (renders['warped'] + 1) / 2
            
            synthesized = renders['synthesized']
            mask = renders['mask']
            fusion_image = fusion_model(synthesized.float(), warped.float(), mask.float())

            warped_pil = to_pil_image(warped[0])
            fusion_pil = to_pil_image(fusion_image[0])

            return warped_pil, fusion_pil

        # Blocks.
        gr.Markdown(
            """
            # StereoGen: Towards Open-World Generation of Stereo Images and Unsupervised Matching
            [![Project Site](https://img.shields.io/badge/Project-Web-green)](https://qjizhi.github.io/genstereo) &nbsp;
            [![Spaces](https://img.shields.io/badge/Spaces-Demo-yellow?logo=huggingface)](https://huggingface.co/spaces/FQiao/GenStereo) &nbsp;
            [![Github](https://img.shields.io/badge/Github-Repo-orange?logo=github)](https://github.com/Qjizhi/GenStereo) &nbsp;
            [![Models](https://img.shields.io/badge/Models-checkpoints-blue?logo=huggingface)](https://huggingface.co/FQiao/GenStereo/tree/main) &nbsp;
            [![arXiv](https://img.shields.io/badge/arXiv-2503.12720-red?logo=arxiv)](https://arxiv.org/abs/2503.12720)
        
            ## Introduction 
            This is an official demo for the paper "[Towards Open-World Generation of Stereo Images and Unsupervised Matching](https://qjizhi.github.io/genstereo)". Given an arbitrary reference image, GenStereo can generate the corresponding right-view image.
        
            ## How to Use

            1. Upload a reference image to "Left Image"
                - You can also select an image from "Examples"
            3. Hit "Generate a right image" button and check the result

            """
        )
        file = gr.File(label='Left', file_types=['image'])
        examples = gr.Examples(
            examples=['./assets/COCO_val2017_000000070229.jpg',
                    './assets/COCO_val2017_000000092839.jpg',
                    './assets/KITTI2015_000003_10.png',
                    './assets/KITTI2015_000147_10.png'],
            inputs=file
        )
        with gr.Row():
            image_widget = gr.Image(
                label='Left Image', type='filepath',
                interactive=False
            )
            depth_widget = gr.Image(label='Estimated Depth', type='pil')

        # Add scale factor slider
        scale_slider = gr.Slider(
            label='Scale Factor',
            minimum=1.0,
            maximum=30.0,
            value=15.0,
            step=0.1,
        )

        button = gr.Button('Generate a right image', size='lg', variant='primary')
        with gr.Row():
            warped_widget = gr.Image(
                label='Warped Image', type='pil', interactive=False
            )
            gen_widget = gr.Image(
                label='Generated Right', type='pil', interactive=False
            )

        # Events
        file.change(
            fn=cb_mde,
            inputs=file,
            outputs=[image_widget, depth_widget, src_image, src_depth]
        )
        button.click(
            fn=cb_generate,
            inputs=[src_image, src_depth, scale_slider],
            outputs=[warped_widget, gen_widget]
        )

    if __name__ == '__main__':
        demo.launch(share=True)