#!/usr/bin/env python3
import os
import sys
import argparse
from os.path import join, basename, splitext

import numpy as np
import cv2
import torch
from PIL import Image
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image
import matplotlib

import lpips
import piq

from genstereo import GenStereo, AdaptiveFusionLayer


# ----------------------------
# Utils
# ----------------------------
def colorize(value, vmin=None, vmax=None, cmap='gray_r', invalid_val=-99,
            invalid_mask=None, background_color=(128, 128, 128, 255),
            gamma_corrected=False, value_transform=None):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    value = value.squeeze()
    if invalid_mask is None:
        invalid_mask = value == invalid_val
    mask = np.logical_not(invalid_mask)

    if mask.any():
        vmin = np.percentile(value[mask], 2) if vmin is None else vmin
        vmax = np.percentile(value[mask], 85) if vmax is None else vmax
    else:
        vmin, vmax = 0.0, 1.0

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value *= 0.0

    value[invalid_mask] = np.nan
    cmapper = matplotlib.cm.get_cmap(cmap)
    if value_transform:
        value = value_transform(value)
    img = cmapper(value, bytes=True)
    img[invalid_mask] = background_color

    if gamma_corrected:
        img = img.astype(np.float32) / 255.0
        img = np.power(img, 2.2)
        img = (img * 255.0).astype(np.uint8)
    return img


def read_pfm(file_path):
    with open(file_path, 'rb') as f:
        header = f.readline().decode('utf-8').rstrip()
        if header not in ('PF', 'Pf'):
            raise Exception('Not a PFM file.')

        color = header == 'PF'
        width, height = map(int, f.readline().decode('utf-8').split())
        scale = float(f.readline().decode('utf-8').rstrip())
        endian = '<' if scale < 0 else '>'
        data = np.fromfile(f, endian + 'f').reshape((height, width, 3) if color else (height, width))
        return np.flipud(data), abs(scale)


def square_center_crop_pil(img: Image.Image) -> Image.Image:
    W, H = img.size
    if W == H:
        return img
    if W < H:
        left, right = 0, W
        top = int(np.ceil((H - W) / 2.0))
        bottom = top + W
    else:
        top, bottom = 0, H
        left = int(np.ceil((W - H) / 2.0))
        right = left + H
    return img.crop((left, top, right, bottom))


def square_center_crop_np(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    if h == w:
        return arr
    size = min(h, w)
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return arr[y0:y0+size, x0:x0+size]


def resize_disp_keep_scale(disparity: np.ndarray, out_size: int) -> np.ndarray:
    h, w = disparity.shape[:2]
    ratio = float(out_size) / float(w)
    disp_resized = cv2.resize(disparity, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
    return disp_resized * ratio


def load_image_for_metric(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
    return img


def compute_metrics(gt_img_path: str, pred_img_path: str, device: torch.device, lpips_fn) -> dict:
    x = torch.tensor(load_image_for_metric(gt_img_path)).permute(2, 0, 1)[None, ...] / 255.0
    y = torch.tensor(load_image_for_metric(pred_img_path)).permute(2, 0, 1)[None, ...] / 255.0

    img0 = lpips.im2tensor(lpips.load_image(gt_img_path))
    img1 = lpips.im2tensor(lpips.load_image(pred_img_path))

    x = x.to(device)
    y = y.to(device)
    img0 = img0.to(device)
    img1 = img1.to(device)

    return {
        "ssim": piq.ssim(x, y, data_range=1.).item(),
        "psnr": piq.psnr(x, y, data_range=1.).mean().item(),
        "lpips": lpips_fn(img0, img1).item()
    }


def morphological_opening(mask_tensor, kernel_size=7):
    mask_np = mask_tensor.squeeze().detach().cpu().numpy()
    mask_np = (mask_np > 0).astype(np.uint8)
    kernel = np.ones((1, 3), np.uint8)
    cleaned = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel)
    return torch.tensor(cleaned, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


# ----------------------------
# Core
# ----------------------------
def build_model(sd_version: str, checkpoint_name: str) -> GenStereo:
    cfg = dict(
        pretrained_model_path='checkpoints',
        checkpoint_name=checkpoint_name,
        half_precision_weights=False,
    )
    return GenStereo(cfg=cfg, sd_version=sd_version)


def infer_image_size(sd_version: str) -> int:
    if sd_version == "v1.5":
        return 512
    if sd_version == "v2.1":
        return 768
    raise ValueError(f"Unknown SD version: {sd_version}")


def load_fusion_model(checkpoint_name: str, device):
    fusion = AdaptiveFusionLayer()
    ckpt_path = os.path.join('checkpoints', checkpoint_name, 'fusion_layer.pth')
    state = torch.load(ckpt_path, map_location='cpu')
    fusion.load_state_dict(state)
    fusion = fusion.to(device)
    fusion.eval()
    return fusion


def run_middlebury(args, model, device, image_size, lpips_fn, fusion_model):
    path_root = args.path_root
    output_root = f'output/middlebury/{args.checkpoint_name}/{ "gt" if args.gt else "pseudo"}'
    os.makedirs(output_root, exist_ok=True)

    scenes = [s for s in os.listdir(path_root) if 'imperfect' not in s]
    metrics_summary = {"ssim": [], "psnr": [], "lpips": []}
    per_scene = {}

    for scene in tqdm(scenes, desc="Middlebury"):
        scene_dir = join(path_root, scene)
        im0_path = join(scene_dir, "im0.png")
        im1_path = join(scene_dir, "im1.png")
        disp_path = join(scene_dir, "disp0.pfm" if args.gt else "disp0.npy")

        if not (os.path.exists(im0_path) and os.path.exists(im1_path) and os.path.exists(disp_path)):
            continue

        if args.gt:
            disp = read_pfm(disp_path)[0].astype(np.float32)
            if disp.ndim == 3:
                disp = disp[..., 0]
        else:
            disp = np.load(disp_path).astype(np.float32)

        disp_resized = resize_disp_keep_scale(disp, image_size)
        disp_tensor = torch.tensor(disp_resized)[None, None].float().to(device)
        disp_vis = to_pil_image(colorize(disp_tensor))

        image_l = Image.open(im0_path).convert('RGB').resize((image_size, image_size), Image.BILINEAR)
        image_r = Image.open(im1_path).convert('RGB').resize((image_size, image_size), Image.BILINEAR)

        out_dir = join(output_root, scene)
        os.makedirs(out_dir, exist_ok=True)

        renders = model(src_image=image_l, src_disparity=disp_tensor, ratio=None)
        warped = (renders['warped'][0] + 1) / 2
        synth = renders['synthesized'][0]

        to_pil_image(warped).save(join(out_dir, 'warped_image.png'))
        to_pil_image(synth).save(join(out_dir, 'synthesized_image.png'))
        disp_vis.save(join(out_dir, 'depth_image.png'))
        image_r.save(join(out_dir, 'input_image_right.png'))

        mask = renders.get('mask', None)
        if mask is None:
            w = warped.unsqueeze(0)
            gray = (0.299*w[0,0] + 0.587*w[0,1] + 0.114*w[0,2]).unsqueeze(0).unsqueeze(0)
            mask = (gray > 0).float()
        mask = morphological_opening(mask).to(device)

        with torch.no_grad():
            fused = fusion_model(synth.unsqueeze(0).float(), warped.unsqueeze(0).float(), mask.float())
        to_pil_image(fused[0]).save(join(out_dir, 'fusion_image.png'))

        m = compute_metrics(join(out_dir, 'input_image_right.png'),
                            join(out_dir, 'fusion_image.png'),
                            device, lpips_fn)
        per_scene[scene] = m
        for k in metrics_summary:
            metrics_summary[k].append(m[k])

    with open(join(output_root, 'metrics_summary.txt'), 'w') as f:
        for k, v in per_scene.items():
            f.write(f"{k}: SSIM={v['ssim']:.4f}, PSNR={v['psnr']:.4f}, LPIPS={v['lpips']:.4f}\n")
        f.write("\n--- Overall Statistics ---\n")
        for metric in ['ssim', 'psnr', 'lpips']:
            arr = np.array(metrics_summary[metric]) if metrics_summary[metric] else np.array([np.nan])
            f.write(f"{metric.upper()}: Mean={np.nanmean(arr):.4f}, "
                    f"Max={np.nanmax(arr):.4f}, Min={np.nanmin(arr):.4f}\n")


def run_kitti15(args, model, device, image_size, lpips_fn, fusion_model):
    path_root = args.path_root
    output_root = f'output/kitti15/{args.checkpoint_name}/{ "gt" if args.gt else "pseudo"}'
    os.makedirs(output_root, exist_ok=True)

    with open(args.image_list_file, 'r') as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    ssim_vals, psnr_vals, lpips_vals = [], [], []
    per_item = {}

    for line in tqdm(lines, desc="KITTI15"):
        left_rel, right_rel, disp_rel = line.split()[:3]
        image_l_path = join(path_root, left_rel)
        image_r_path = join(path_root, right_rel)
        disp_path = join(path_root, disp_rel)

        if not args.gt:
            disp_path = disp_path.replace("disp_occ_0", "disp_occ_0-pseudo").replace("png", "npy")

        base_name = splitext(basename(image_l_path))[0]
        if args.gt:
            disp = np.array(Image.open(disp_path), dtype=np.float32) / 256.0
        else:
            disp = np.load(disp_path).astype(np.float32)

        disp = square_center_crop_np(disp)
        ratio = float(image_size) / float(disp.shape[1])
        disp_resized = cv2.resize(disp, (image_size, image_size), interpolation=cv2.INTER_NEAREST) * ratio

        disp_tensor = torch.tensor(disp_resized)[None, None].float().to(device)
        disp_vis = to_pil_image(colorize(disp_tensor))

        image_l = square_center_crop_pil(Image.open(image_l_path).convert('RGB')).resize((image_size, image_size), Image.BILINEAR)
        image_r = square_center_crop_pil(Image.open(image_r_path).convert('RGB')).resize((image_size, image_size), Image.BILINEAR)

        out_dir = join(output_root, base_name)
        os.makedirs(out_dir, exist_ok=True)

        renders = model(src_image=image_l, src_disparity=disp_tensor, ratio=None)
        warped = (renders['warped'][0] + 1) / 2
        synth = renders['synthesized'][0]

        image_l.save(join(out_dir, 'input_image.png'))
        to_pil_image(warped).save(join(out_dir, 'warped_image.png'))
        to_pil_image(synth).save(join(out_dir, 'synthesized_image.png'))
        disp_vis.save(join(out_dir, 'depth_image.png'))
        image_r.save(join(out_dir, 'input_image_right.png'))

        mask = renders.get('mask', None)
        if mask is None:
            w = warped.unsqueeze(0)
            gray = (0.299*w[0,0] + 0.587*w[0,1] + 0.114*w[0,2]).unsqueeze(0).unsqueeze(0)
            mask = (gray > 0).float()
        mask = morphological_opening(mask).to(device)

        with torch.no_grad():
            fused = fusion_model(synth.unsqueeze(0).float(), warped.unsqueeze(0).float(), mask.float())
        to_pil_image(fused[0]).save(join(out_dir, 'fusion_image.png'))

        m = compute_metrics(join(out_dir, 'input_image_right.png'),
                            join(out_dir, 'fusion_image.png'),
                            device, lpips_fn)
        per_item[base_name] = m
        ssim_vals.append(m['ssim'])
        psnr_vals.append(m['psnr'])
        lpips_vals.append(m['lpips'])

    with open(join(output_root, 'metrics_summary.txt'), 'w') as f:
        for k, v in per_item.items():
            f.write(f"{k}: SSIM={v['ssim']:.4f}, PSNR={v['psnr']:.4f}, LPIPS={v['lpips']:.4f}\n")
        f.write("\n--- Overall Statistics ---\n")
        f.write(f"SSIM: Mean={np.mean(ssim_vals):.4f}\n")
        f.write(f"PSNR: Mean={np.mean(psnr_vals):.4f}\n")
        f.write(f"LPIPS: Mean={np.mean(lpips_vals):.4f}\n")


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Unified Middlebury/KITTI15 evaluation for GenStereo")
    p.add_argument('--dataset', choices=['middlebury', 'kitti15'], required=True)
    p.add_argument('--path-root', required=True)
    p.add_argument('--image-list-file', default='./data/kitti/kitti15_train200.txt', help='Required for KITTI15')
    p.add_argument('--gt', default=False, action='store_true')
    p.add_argument('--sd-version', choices=['v1.5', 'v2.1'], default='v2.1')
    p.add_argument('--device', choices=['auto', 'cuda', 'mps', 'cpu'], default='auto')
    return p.parse_args()


def main():
    args = parse_args()
    args.checkpoint_name = 'genstereo-v1.5' if args.sd_version == 'v1.5' else 'genstereo-v2.1'
    img_size = infer_image_size(args.sd_version)

    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    lpips_fn = lpips.LPIPS(net='squeeze').to(device)
    model = build_model(args.sd_version, args.checkpoint_name)
    fusion_model = load_fusion_model(args.checkpoint_name, device)

    if args.dataset == 'middlebury':
        run_middlebury(args, model, device, img_size, lpips_fn, fusion_model)
    else:
        if not args.image_list_file:
            print("Error: --image-list-file is required for dataset=kitti15", file=sys.stderr)
            sys.exit(1)
        run_kitti15(args, model, device, img_size, lpips_fn, fusion_model)


if __name__ == '__main__':
    main()
