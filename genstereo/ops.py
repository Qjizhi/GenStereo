import numpy as np
import torch
from torch import Tensor

def convert_left_to_right(left_embed, disparity, left_image, random_ratio=None):
    # Get the height, width, and channels from the left embedding
    _, height, width = left_embed.shape

    # Initialize tensors for right_embed, converted_right_image, and mask
    # right_embed = torch.full_like(left_embed, 255)
    # converted_right_image = torch.full_like(left_image, 255)
    right_embed = torch.ones_like(left_embed)
    converted_right_image = torch.ones_like(left_image)    
    mask = torch.ones((height, width), dtype=torch.uint8, device=left_embed.device)

    # Round the disparity and convert to int
    disparity_rounded = torch.round(disparity).squeeze(0).long()  # [h, w]

    # Loop through the image dimensions and apply the conversion
    for y in range(height):
        for x in range(width):
            new_x = x - disparity_rounded[y, x]

            if 0 <= new_x < width:# and disparity_rounded[y, x] > 0:
                right_embed[:, y, new_x] = left_embed[:, y, x]
                converted_right_image[:, y, new_x] = left_image[:, y, x]
                mask[y, new_x] = 0  # Mark as valid in the mask
    # print(f"Mask sum before: {mask.sum()}")
    # Apply random mask if drop_ratio is set
    if random_ratio is not None:
        print(f"Random ratio: {random_ratio}")
        # Create a random mask with values ranging from 0 (invalid) to 1 (valid)
        random_mask = torch.bernoulli(torch.full((height, width), 1 - random_ratio, device=left_embed.device)).byte()
        # Perform a logical AND operation with the mask from the function
        mask = mask | random_mask

        # Apply the final mask to right_embed, converted_right_image, and disparity
        right_embed[:, mask == 1] = 255  # Set masked out locations to 255 in the right embed
        converted_right_image[:, mask == 1] = 255  # Set masked out locations to 255 in the converted right image
        disparity[:, mask == 1] = 0  # Set masked out locations in the original disparity to 0
        # print(f"Mask sum after: {mask.sum()}")
    return right_embed, mask, converted_right_image, disparity


def convert_left_to_right_torch(left_embed, disparity, left_image, random_ratio=None):
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

        valid_indices = (new_x >= 0) & (new_x < width) #& (disparity_rounded[:, x] > 0)
        valid_new_x = new_x[valid_indices]
        valid_y = torch.arange(height, device=left_embed.device)[valid_indices]

        right_embed[:, valid_y, valid_new_x] = left_embed[:, valid_y, x]
        converted_right_image[:, valid_y, valid_new_x] = left_image[:, valid_y, x]
        mask[valid_y, valid_new_x] = 0  # Mark as valid in the mask

    return right_embed, mask, converted_right_image, disparity