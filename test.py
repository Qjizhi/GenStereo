from os.path import basename, splitext, join
import numpy as np
import torch
import cv2
from torchvision.transforms.functional import to_tensor, to_pil_image
import ssl
import os
from extern.DAM2.depth_anything_v2.dpt import DepthAnythingV2
ssl._create_default_https_context = ssl._create_unverified_context
from PIL import Image
import argparse

from genstereo import GenStereo, AdaptiveFusionLayer

SD_VERSION = "v2.1"
if SD_VERSION == "v1.5":
    IMAGE_SIZE = 512
    CHECKPOINT_NAME = 'genstereo-v1.5'
elif SD_VERSION == "v2.1":
    IMAGE_SIZE = 768
    CHECKPOINT_NAME = 'genstereo-v2.1'
else:
    raise ValueError(f"Unknown SD version: {SD_VERSION}")

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

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
size_name = encoder_size_map[encoder]
dam2_path = f"https://huggingface.co/depth-anything/Depth-Anything-V2-{size_name}/resolve/main/depth_anything_v2_{encoder}.pth"

checkpoint_dir = 'checkpoints'
dam2_checkpoint = f'{checkpoint_dir}/depth_anything_v2_{encoder}.pth'
os.makedirs(checkpoint_dir, exist_ok=True)

if not os.path.exists(dam2_checkpoint):
    print(f"Downloading DAM2 model from {dam2_path}")
    os.system(f"wget {dam2_path} -O {dam2_checkpoint}")

dam2.load_state_dict(torch.load(dam2_checkpoint, map_location='cpu'))
dam2 = dam2.to(DEVICE).eval()

genstereo_cfg = dict(
    pretrained_model_path=checkpoint_dir,
    checkpoint_name=CHECKPOINT_NAME,
    half_precision_weights=True if 'cuda' in DEVICE else False,
)
genstereo_nvs = GenStereo(cfg=genstereo_cfg, device=DEVICE, sd_version=SD_VERSION)

fusion_model = AdaptiveFusionLayer()
fusion_checkpoint = join(checkpoint_dir, CHECKPOINT_NAME, 'fusion_layer.pth')
fusion_model.load_state_dict(torch.load(fusion_checkpoint))
fusion_model = fusion_model.to(DEVICE).eval()

def crop(img: Image) -> Image:
    W, H = img.size
    if W < H:
        crop_size = W
        top = (H - crop_size) // 2
        bottom = top + crop_size
        left, right = 0, W
    else:
        crop_size = H
        left = (W - crop_size) // 2
        right = left + crop_size
        top, bottom = 0, H
    return img.crop((left, top, right, bottom))

def infer_depth_dam2(image_path: str):
    image = Image.open(image_path).convert('RGB')
    image = crop(image).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    depth_dam2 = dam2.infer_image(image_bgr)
    return image, torch.tensor(depth_dam2).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

def normalize_disp(disp):
    return (disp - disp.min()) / (disp.max() - disp.min())

def morphological_opening(mask_tensor, kernel_size=7):
    mask_np = mask_tensor.squeeze().cpu().numpy().astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cleaned_mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel)
    return torch.tensor(cleaned_mask_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

def generate_novel_view(image, depth, output_dir, basename, scale_factor=0.15):
    disparity = normalize_disp(depth) * scale_factor * IMAGE_SIZE
    output_path = join(output_dir, basename)
    os.makedirs(output_path, exist_ok=True)

    renders = genstereo_nvs(src_image=image, src_disparity=disparity, ratio=None)
    warped = (renders['warped'] + 1) / 2
    mask = morphological_opening(renders['mask'])
    
    with torch.no_grad():
        fusion_image = fusion_model(renders['synthesized'].float(), warped.float(), mask.float())
    
    warped_pil = to_pil_image(warped[0])
    fusion_image_pil = to_pil_image(fusion_image[0])
    disp_vis = cv2.applyColorMap((normalize_disp(depth.squeeze().cpu().numpy()) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    
    cv2.imwrite(join(output_path, 'disp.png'), disp_vis)
    image.save(join(output_path, 'left.png'))
    warped_pil.save(join(output_path, 'warped.png'))
    fusion_image_pil.save(join(output_path, 'generated_right.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate novel view from input image")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--output", default="./vis", help="Output directory")
    parser.add_argument("--scale_factor", type=float, default=0.15, help="Disparity scaling factor")
    args = parser.parse_args()
    
    base_name = splitext(basename(args.image_path))[0]
    img, depth = infer_depth_dam2(args.image_path)
    generate_novel_view(img, depth, args.output, base_name, args.scale_factor)