import json
import random

import torch
import torchvision.transforms as transforms
# from decord import VideoReader
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor
from jaxtyping import Float
import h5py


import os
import json
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import cv2


def convert_left_to_right_torch(left_embed, disparity, left_image, random_ratio=None, dataset_name=None):
    """
    Convert left features to right features based on disparity values.
    
    Args:
        left_embed (torch.Tensor): [c, h, w] tensor representing left feature embeddings.
        disparity (torch.Tensor): [1, h, w] tensor of disparity values.
        left_image (torch.Tensor): [3, h, w] tensor representing the left image.

    Returns:
        right_embed (torch.Tensor): [c, h, w] tensor for the right feature embeddings.
        mask (torch.Tensor): [h, w] binary mask (1 = invalid, 0 = valid).
        converted_right_image (torch.Tensor): [3, h, w] tensor for the right image.
        disparity (torch.Tensor): [1, h, w] tensor for the disparity.
    """
    # Get the height, width, and channels from the left embedding
    _, height, width = left_embed.shape

    # Initialize tensors for right_embed, converted_right_image, and mask
    right_embed = torch.zeros_like(left_embed)
    # converted_right_image = torch.zeros_like(left_image)
    converted_right_image = -torch.ones_like(left_image)
    mask = torch.ones((height, width), dtype=torch.uint8, device=left_embed.device)

    # Round the disparity and convert to int
    disparity_rounded = torch.round(disparity).squeeze(0).long()  # [h, w]

    # Iterate over width and process each column for all rows
    for x in range(width):
        new_x = x - disparity_rounded[:, x]
        if dataset_name == 'InStereo2K' or dataset_name == 'DrivingStereo':
            valid_indices = (new_x >= 0) & (new_x < width) & (disparity_rounded[:, x] > 0)
        else:
            valid_indices = (new_x >= 0) & (new_x < width)
        valid_new_x = new_x[valid_indices]
        valid_y = torch.arange(height, device=left_embed.device)[valid_indices]

        right_embed[:, valid_y, valid_new_x] = left_embed[:, valid_y, x]
        converted_right_image[:, valid_y, valid_new_x] = left_image[:, valid_y, x]
        mask[valid_y, valid_new_x] = 0  # Mark as valid in the mask
    # Apply random masking if random_ratio is set
    if random_ratio is not None:
        # Create a random mask
        random_mask = torch.bernoulli(torch.full((height, width), 1 - random_ratio, device=left_embed.device)).byte()
        mask |= random_mask

        # Apply the mask to right_embed, converted_right_image, and disparity
        right_embed[:, mask == 1] = 0  # Mask out invalid regions in right_embed
        converted_right_image[:, mask == 1] = 0  # Mask out invalid regions in converted_right_image
        disparity[:, mask == 1] = 0  # Mask out invalid regions in disparity

    return right_embed, mask, converted_right_image, disparity

class StereoGenDataset(Dataset):
    def __init__(self,
        json_files,
        img_size,
        img_scale=(1.0, 1.0),
        img_ratio=(0.9, 1.0),
        drop_ratio=0.1,
        debug=False,
        use_coords=True,
        use_wapred=True,):
        """
        Args:
            json_files (list): Paths to the JSON file.
        """
        super().__init__()

        self.data = []
        for json_file in json_files:
            with open(json_file, 'r') as f:
                previous_length = len(self.data)
                self.data += json.load(f)
                added_length = len(self.data) - previous_length
                print(f"Loaded {added_length} samples from {json_file}")
        self.img_size = img_size

        self.embedder = self.get_embedder(2)
        self.drop_ratio = drop_ratio

        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Convert PIL image to Tensor and scale to [0, 1]
        ])

        self.transform_pixels = transforms.Compose([
            transforms.ToTensor(),  # Converts image to Tensor
            transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
        ])
        self.clip_image_processor = CLIPImageProcessor()
        self.debug = debug
        self.use_coords = use_coords
        self.use_wapred = use_wapred

    def __len__(self):
        return len(self.data)

    def crop(self, img: Image) -> Image:
        W, H = img.size
        if W < H:
            left, right = 0, W
            top, bottom = np.ceil((H - W) / 2.), np.floor((H - W) / 2.) + W
        else:
            left, right = np.ceil((W - H) / 2.), np.floor((W - H) / 2.) + H
            top, bottom = 0, H
        img = img.crop((left, top, right, bottom))
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        return img

    def crop_and_resize_disp(self, disparity_left):
        # Determine the smaller side
        h, w = disparity_left.shape[:2]
        min_side = min(h, w)
        
        # Calculate the cropping coordinates
        start_x = (w - min_side) // 2
        start_y = (h - min_side) // 2
        
        # Crop the array to a square
        cropped_disparity = disparity_left[start_y:start_y + min_side, start_x:start_x + min_side]

        # Resize the cropped array to the desired size
        ratio = self.img_size / min_side
        resized_disparity = cv2.resize(cropped_disparity, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR) * ratio
        
        return resized_disparity
    
    def random_crop_and_resize(self, image_left: Image, image_right: Image, disparity_left: np.ndarray):
        """
        Randomly crop and resize stereo image pairs and their disparity maps.

        Args:
            image_left (Image.Image): Left image (PIL).
            image_right (Image.Image): Right image (PIL).
            disparity_left (np.ndarray): Left disparity map.

        Returns:
            tuple: Resized left image, right image, and disparity map.
        """
        # Get the dimensions of the image and disparity map
        W, H = image_left.size
        h_disp, w_disp = disparity_left.shape[:2]

        # Ensure the image and disparity map have the same dimensions
        assert W == w_disp and H == h_disp, "Image and disparity dimensions must match."
        assert isinstance(image_left, Image.Image) and isinstance(image_right, Image.Image), \
            "Inputs must be PIL images."
        assert isinstance(disparity_left, np.ndarray), "Disparity must be a NumPy array."

        # Determine crop size
        if min(W, H) > 3 * self.img_size:
            crop_size = 3 * self.img_size
        elif min(W, H) > 2 * self.img_size:
            crop_size = 2 * self.img_size
        elif min(W, H) >= self.img_size:
            crop_size = self.img_size
        else:
            crop_size = min(W, H)

        # Calculate random crop coordinates
        max_x = W - crop_size
        max_y = H - crop_size
        left = random.randint(0, max(max_x, 0))
        top = random.randint(0, max(max_y, 0))
        right = left + crop_size
        bottom = top + crop_size

        # Perform cropping
        image_left_cropped = image_left.crop((left, top, right, bottom))
        image_right_cropped = image_right.crop((left, top, right, bottom))
        disparity_cropped = disparity_left[top:bottom, left:right]

        # Resize images and disparity map if necessary
        if crop_size != self.img_size:
            image_left_resized = image_left_cropped.resize((self.img_size, self.img_size), Image.BILINEAR)
            image_right_resized = image_right_cropped.resize((self.img_size, self.img_size), Image.BILINEAR)
            ratio = self.img_size / crop_size
            disparity_resized = cv2.resize(disparity_cropped, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR) * ratio
        else:
            image_left_resized = image_left_cropped
            image_right_resized = image_right_cropped
            disparity_resized = disparity_cropped

        return image_left_resized, image_right_resized, disparity_resized

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

    def getdata(self, idx):
        try:
            image_left_path = self.data[idx]['image_left']
            image_right_path = self.data[idx]['image_right']
            image_left = Image.open(image_left_path).convert('RGB')
            image_right = Image.open(image_right_path).convert('RGB')
            disparity_left = None
            dataset_name = self.data[idx]["dataset"]
            if dataset_name == 'VKITTI2':
                depth_left_path = self.data[idx]['depth_left']
                depth_left = cv2.imread(depth_left_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR) / 100.
                # invalid = depth_left >= 65535
                disparity_left = 0.532725 * 725.0087 / (depth_left + 1e-5)  # f = 725.0087, b = 0.532725 # meter             
            else:
                print(f"Dataset {self.data[idx]['dataset']} is not supported.")
            return image_left, image_right, disparity_left, dataset_name
        except Exception as e:
            bad_file_path = self.data[idx]['image_left']  # Capture the bad file path
            print(f"Error loading data from {bad_file_path}: {e}")
            return None, None, None, None

    def __getitem__(self, idx):
        # 1. Load images and depth maps
        image_left, image_right, disparity_left, dataset_name = self.getdata(idx)

        # Retry or skip sample if None is returned
        if image_left is None or image_right is None or disparity_left is None:
            print(f"Data at index {idx} is invalid. Skipping.")
            return self.__getitem__((idx + 1) % len(self.data))  # Try next index

        # 2. Crop and resize
        image_left, image_right, disparity_left = self.random_crop_and_resize(image_left, image_right, disparity_left)

        # 3. Generate coords
        grid: Float[Tensor, 'H W C'] = torch.stack(torch.meshgrid(
            torch.arange(self.img_size), torch.arange(self.img_size), indexing='xy'), dim=-1
        )

        # 4. Coordinates embedding.
        coords = torch.stack((grid[..., 0]/self.img_size, grid[..., 1]/self.img_size), dim=-1)
        embed = self.embedder(coords)
        embed = embed.permute(2, 0, 1)

        # 5. Convert to PyTorch tensors
        image_left_tensor = self.transform_pixels(image_left)
        image_right_tensor = self.transform_pixels(image_right)  
        disparity_left_tensor = torch.tensor(disparity_left, dtype=torch.float32).unsqueeze(0)  # Add a channel dimension

        # 6. Warp left to right
        random_mask = random.random()
        rando_ratio = random.random() if random_mask < self.drop_ratio else None
        warped_embed, mask, converted_right, disparity_left_tensor = convert_left_to_right_torch(embed, disparity_left_tensor, image_left_tensor, rando_ratio, dataset_name=dataset_name)
        if self.debug:
            save_folder = "./check_dataset/"
            os.makedirs(save_folder, exist_ok=True)
            # cv2.imwrite(f"{save_folder}/{dataset_name}_{idx}_left.png", (image_left_tensor.permute(1, 2, 0).numpy()[:, :, ::-1]/2+0.5)*255)
            cv2.imwrite(f"{save_folder}/{dataset_name}_{idx}_mask.png", mask.numpy()*255)
            cv2.imwrite(f"{save_folder}/{dataset_name}_{idx}_right.png", (image_right_tensor.permute(1, 2, 0).numpy()[:, :, ::-1]/2+0.5)*255)
            cv2.imwrite(f"{save_folder}/{dataset_name}_{idx}_converted_right.png", (converted_right.permute(1, 2, 0).numpy()[:, :, ::-1]/2+0.5)*255)

        # 7. Add mask to the embeddings
        if self.use_coords and self.use_wapred:
            src_coords_embed = torch.cat(
                [embed, torch.zeros_like(mask.unsqueeze(0), device=mask.device), image_left_tensor], dim=0)
            trg_coords_embed = torch.cat([warped_embed, mask.unsqueeze(0), converted_right], dim=0)
        elif self.use_coords and not self.use_wapred:
            src_coords_embed = torch.cat([embed, torch.zeros_like(mask.unsqueeze(0), device=mask.device)], dim=0)
            trg_coords_embed = torch.cat([warped_embed, mask.unsqueeze(0)], dim=0)
        else:
            src_coords_embed = torch.cat([image_left_tensor, torch.zeros_like(mask.unsqueeze(0), device=mask.device)], dim=0)
            trg_coords_embed = torch.cat([converted_right, mask.unsqueeze(0)], dim=0)

        # 8. Get clip image
        clip_image = self.clip_image_processor(
            images=image_left, return_tensors="pt"
        ).pixel_values[0]

        sample = {
            'source': image_left_tensor,
            'correspondence': disparity_left_tensor,
            'target': image_right_tensor,
            'src_coords_embed': src_coords_embed,
            'trg_coords_embed': trg_coords_embed,
            'clip_images':clip_image,
            'converted_right': converted_right,
            'mask': mask.unsqueeze(0),
        }

        return sample


if __name__ == "__main__":
    # Load the dataset from JSON file
    json_file = [
        "./data/VKITTI2/VKITTI2_dataset_paths_2.json",
        ]

    dataset = StereoGenDataset(json_file, img_size=768)
    print(f"Number of entries in the dataset: {len(dataset)}")

    # Sample 20 random entries
    sampled_indices = random.sample(range(len(dataset)), 20)

    # Loop through the sampled indices and access dataset entries
    for idx in sampled_indices:
        data_entry = dataset[idx]
        # You can now do something with data_entry, e.g., printing or processing it
        print(f"Processing dataset entry at index {idx}")
    # dataset.__getitem__(1000)

