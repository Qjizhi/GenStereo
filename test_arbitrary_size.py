from os.path import basename, splitext, join
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from torchvision.transforms.functional import to_tensor, to_pil_image
import ssl
import os
from extern.DAM2.depth_anything_v2.dpt import DepthAnythingV2
ssl._create_default_https_context = ssl._create_unverified_context
from PIL import Image
import argparse
from tqdm import tqdm

from genwarp import GenWarp

from train import AdaptiveFusionLayer

# --- Constants and Model Setup ---
PATCH_SIZE = 768 # Using a constant for patch size
CHECKPOINT_NAME = 'mar24_1_all_alpha1_sd2-1/epoch5'
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# --- Model Configurations ---
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}

encoder = 'vitl'
encoder_size_map = {'vits': 'Small', 'vitb': 'Base', 'vitl': 'Large'}

if encoder not in encoder_size_map:
    raise ValueError(f"Unsupported encoder: {encoder}. Supported: {list(encoder_size_map.keys())}")

# --- Load Depth Anything V2 Model ---
print("Loading Depth Anything V2 model...")
dam2 = DepthAnythingV2(**model_configs[encoder])
size_name = encoder_size_map[encoder]
dam2_path = f"https://huggingface.co/depth-anything/Depth-Anything-V2-{size_name}/resolve/main/depth_anything_v2_{encoder}.pth"

checkpoint_dir = 'checkpoints'
dam2_checkpoint = f'{checkpoint_dir}/depth_anything_v2_{encoder}.pth'
os.makedirs(checkpoint_dir, exist_ok=True)

if not os.path.exists(dam2_checkpoint):
    print(f"Downloading DAM2 model from {dam2_path}...")
    os.system(f"wget {dam2_path} -O {dam2_checkpoint}")

dam2.load_state_dict(torch.load(dam2_checkpoint, map_location='cpu'))
dam2 = dam2.to(DEVICE).eval()
print("Depth model loaded.")

# --- Load GenWarp and Fusion Models ---
print("Loading GenWarp and Fusion models...")
genstereo_cfg = dict(
    pretrained_model_path=checkpoint_dir,
    checkpoint_name=CHECKPOINT_NAME,
    half_precision_weights=False, #True if 'cuda' in DEVICE else False,
)
genstereo_nvs = GenWarp(cfg=genstereo_cfg, device=DEVICE)

fusion_model = AdaptiveFusionLayer()
fusion_checkpoint = join(checkpoint_dir, CHECKPOINT_NAME, 'fusion_layer.pth')
fusion_model.load_state_dict(torch.load(fusion_checkpoint))
fusion_model = fusion_model.to(DEVICE).eval()
print("All models loaded successfully.")

# --- Image and Utility Functions ---

def infer_depth_dam2(image: Image):
    """
    Infers depth for the given PIL image.
    """
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    depth_dam2 = dam2.infer_image(image_bgr) 
    return torch.tensor(depth_dam2).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

def normalize_depth(depth):
    """
    Normalizes a depth tensor to the range [0, 1] based on its global min/max.
    """
    return (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

def morphological_opening(mask_tensor, kernel_size=7):
    """
    Applies morphological opening to a mask tensor to remove small noise.
    """
    mask_np = mask_tensor.squeeze().cpu().numpy().astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cleaned_mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel)
    return torch.tensor(cleaned_mask_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

def find_seam_and_stitch(im1, im2, stride, patch_size, orientation='horizontal'):
    """
    Finds the optimal seam in the overlapping region of two images and stitches them.
    A seam is a straight line (vertical or horizontal) where the L1 error is minimal.
    """
    if orientation == 'horizontal':
        # im1 is the stitched image so far (left), im2 is the new patch (right)
        overlap_width = im1.shape[3] + patch_size - (im1.shape[3] // stride + 1) * stride
        
        if overlap_width <= 0: return torch.cat([im1, im2], dim=3)

        # Extract overlapping regions
        overlap1 = im1[:, :, :, -overlap_width:]
        overlap2 = im2[:, :, :, :overlap_width]

        # Calculate L1 error for each column in the overlap and find the best seam
        l1_errors = torch.abs(overlap1 - overlap2).sum(dim=(0, 1, 2)) # Sum over C, H
        seam_index_in_overlap = torch.argmin(l1_errors).item()

        # Stitch the images along the optimal seam
        left_part = im1[:, :, :, :im1.shape[3] - overlap_width + seam_index_in_overlap]
        right_part = im2[:, :, :, seam_index_in_overlap:]
        return torch.cat([left_part, right_part], dim=3)
    
    else: # vertical
        # im1 is the stitched rows so far (top), im2 is the new row (bottom)
        overlap_height = im1.shape[2] + patch_size - (im1.shape[2] // stride + 1) * stride
        
        if overlap_height <= 0: return torch.cat([im1, im2], dim=2)

        # Extract overlapping regions
        overlap1 = im1[:, :, -overlap_height:, :]
        overlap2 = im2[:, :, :overlap_height, :]

        # Calculate L1 error for each row in the overlap and find the best seam
        l1_errors = torch.abs(overlap1 - overlap2).sum(dim=(0, 1, 3)) # Sum over C, W
        seam_index_in_overlap = torch.argmin(l1_errors).item()

        # Stitch the images along the optimal seam
        top_part = im1[:, :, :im1.shape[2] - overlap_height + seam_index_in_overlap, :]
        bottom_part = im2[:, :, seam_index_in_overlap:, :]
        return torch.cat([top_part, bottom_part], dim=2)

# --- Main Processing Logic ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a novel view from an input image with overlapping patches.")
    parser.add_argument("image_path", help="Path to the input image.")
    parser.add_argument("--output", default="./vis_arbitrary_res_new", help="Output directory.")
    parser.add_argument("--scale_factor", type=float, default=0.10, help="Disparity scaling factor.")
    parser.add_argument("--overlap_ratio", type=float, default=0.5, help="Patch overlap ratio (0.0 to 1.0). Default: 0.5 (50%%).")
    args = parser.parse_args()

    if not (0.0 <= args.overlap_ratio < 1.0):
        raise ValueError("Overlap ratio must be between 0.0 and 1.0 (exclusive of 1.0).")

    # --- 1. Load image and prepare dimensions ---
    print(f"Processing {args.image_path}...")
    img = Image.open(args.image_path).convert('RGB')
    original_W, original_H = img.size
    print(f"Original size: {original_W}x{original_H}")

    resized_W = ((original_W + PATCH_SIZE - 1) // PATCH_SIZE) * PATCH_SIZE
    resized_H = ((original_H + PATCH_SIZE - 1) // PATCH_SIZE) * PATCH_SIZE
    
    print(f"Resizing to a multiple of {PATCH_SIZE}: {resized_W}x{resized_H} for processing...")
    img_resized = img.resize((resized_W, resized_H), Image.LANCZOS)
    
    # --- 2. Inference depth and calculate disparity for the ENTIRE image ---
    full_depth = infer_depth_dam2(img_resized)
    
    print("Normalizing full depth map and calculating disparity...")
    full_depth_normalized = normalize_depth(full_depth)
    full_disparity = full_depth_normalized * args.scale_factor * PATCH_SIZE

    # --- 3. Save visualizations and create patches with overlap ---
    base_name = splitext(basename(args.image_path))[0]
    output_path = join(args.output, base_name)
    os.makedirs(output_path, exist_ok=True)
    
    img_resized.save(join(output_path, 'input_left_resized.png'))
    depth_vis = cv2.applyColorMap((full_depth_normalized.squeeze().cpu().numpy() * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    cv2.imwrite(join(output_path, 'full_depth_map.png'), depth_vis)

    # Calculate stride based on the desired overlap
    stride = int(PATCH_SIZE * (1 - args.overlap_ratio))
    print(f"Using patch size {PATCH_SIZE}x{PATCH_SIZE} with a stride of {stride} pixels ({args.overlap_ratio*100:.0f}% overlap).")

    # Patch the source image and the full disparity map
    img_tensor = to_tensor(img_resized).unsqueeze(0).to(DEVICE)
    img_patches = F.unfold(img_tensor, kernel_size=PATCH_SIZE, stride=stride)
    disparity_patches = F.unfold(full_disparity, kernel_size=PATCH_SIZE, stride=stride)

    # Reshape unfolded tensors to a batch of patches
    num_patches = img_patches.shape[-1]
    img_patches = img_patches.permute(2, 0, 1).reshape(num_patches, 3, PATCH_SIZE, PATCH_SIZE)
    disparity_patches = disparity_patches.permute(2, 0, 1).reshape(num_patches, 1, PATCH_SIZE, PATCH_SIZE)

    # --- 4. Process each patch with its pre-calculated disparity ---
    generated_patches = []
    print(f"Generating novel views for {num_patches} patches...")
    for i in tqdm(range(num_patches)):
        img_patch_pil = to_pil_image(img_patches[i].cpu())
        disparity_patch = disparity_patches[i].unsqueeze(0)
        
        with torch.no_grad():
            renders = genstereo_nvs(src_image=img_patch_pil, src_disparity=disparity_patch, ratio=None)
            warped = (renders['warped'] + 1) / 2
            mask = morphological_opening(renders['mask'])
            fusion_image = fusion_model(renders['synthesized'].float(), warped.float(), mask.float())
            generated_patches.append(fusion_image)

    # --- 5. Merge generated patches using optimal seam stitching ---
    print("Merging patches with optimal seam stitching...")

    # Calculate grid dimensions of patches
    num_patches_H = (resized_H - PATCH_SIZE) // stride + 1 if resized_H >= PATCH_SIZE else 1
    num_patches_W = (resized_W - PATCH_SIZE) // stride + 1 if resized_W >= PATCH_SIZE else 1

    if num_patches != num_patches_H * num_patches_W:
        raise ValueError("Mismatch between total patches and grid dimensions. Check unfolding logic.")

    # Convert the flat list of generated patches into a 2D grid for easier handling
    patch_grid = [generated_patches[i*num_patches_W:(i+1)*num_patches_W] for i in range(num_patches_H)]

    # First, stitch all patches in each row horizontally
    stitched_rows = []
    for r in tqdm(range(num_patches_H), desc="Stitching Rows"):
        # Start with the first patch in the row
        current_row_image = patch_grid[r][0]
        for c in range(1, num_patches_W):
            right_patch = patch_grid[r][c]
            # The find_seam_and_stitch function is now stateful on the growing stitched image
            current_row_image = find_seam_and_stitch(current_row_image, right_patch, stride, PATCH_SIZE, orientation='horizontal')
        stitched_rows.append(current_row_image)

    # Second, stitch the completed rows together vertically
    if not stitched_rows:
        raise ValueError("No patches were generated to stitch.")

    reconstructed_image = stitched_rows[0]
    for r in tqdm(range(1, num_patches_H), desc="Stitching Final Image"):
        bottom_row = stitched_rows[r]
        reconstructed_image = find_seam_and_stitch(reconstructed_image, bottom_row, stride, PATCH_SIZE, orientation='vertical')
    
    # --- 6. Resize the final image back to original dimensions ---
    reconstructed_image_pil = to_pil_image(reconstructed_image.squeeze(0).cpu())
    
    print(f"Resizing final image back to original size: {original_W}x{original_H}")
    final_image = reconstructed_image_pil.resize((original_W, original_H), Image.LANCZOS)
    
    output_filename = join(output_path, f'generated_right_sf{args.scale_factor}_ov{args.overlap_ratio}_seam.png')
    final_image.save(output_filename)

    print(f"Processing complete. Final image saved to {output_filename}")
