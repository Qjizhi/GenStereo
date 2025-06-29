import os
import json
from tqdm import tqdm 

def generate_vkitti_dataset_json(root_path, output_file):
    dataset = []
    num=0
    # Traverse the directory structure
    for scene in os.listdir(root_path):
        scene_path = os.path.join(root_path, scene)
        if os.path.isdir(scene_path):
            for condition in os.listdir(scene_path):  # E.g., overcast, clone, etc.
                condition_path = os.path.join(scene_path, condition)
                if os.path.isdir(condition_path):
                    rgb_camera_0_path = os.path.join(condition_path, "frames", "rgb", "Camera_0")
                    rgb_camera_1_path = os.path.join(condition_path, "frames", "rgb", "Camera_1")
                    depth_camera_0_path = os.path.join(condition_path, "frames", "depth", "Camera_0")
                    
                    if os.path.isdir(rgb_camera_0_path) and os.path.isdir(rgb_camera_1_path) and os.path.isdir(depth_camera_0_path):
                        for file_name in os.listdir(rgb_camera_0_path):
                            if file_name.endswith(".jpg"):
                                base_name = file_name.split(".jpg")[0]

                                # Paths for left image, right image, and left depth
                                image_left_path = os.path.join(rgb_camera_0_path, file_name)
                                image_right_path = os.path.join(rgb_camera_1_path, f"{base_name}.jpg")
                                depth_left_path = os.path.join(depth_camera_0_path, f"{base_name.replace('rgb', 'depth')}.png")

                                # Add this entry to the dataset list
                                dataset.append({
                                    "dataset": "VKITTI2",
                                    "image_left": image_left_path,
                                    "image_right": image_right_path,
                                    "depth_left": depth_left_path,
                                })
                                num+=1
    print("Tatoal number of images: ", num)
    # Duplicate the dataset for training because VKITTI2 doesn't have a enough samples.
    duplicated_dataset = dataset * 2

    # Write the dataset information to a JSON file
    with open(output_file, 'w') as f:
        json.dump(duplicated_dataset, f, indent=4)

# Example usage:
root_path = "/path/to/VKITTI2"
output_file = "VKITTI2_dataset_paths_2.json"
generate_vkitti_dataset_json(root_path, output_file)
