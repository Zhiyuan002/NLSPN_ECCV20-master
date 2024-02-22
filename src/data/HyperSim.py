import os
import random

import torch
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
from PIL import Image
import json
import cv2
def visualize_tensor(img, mode, window="Image", depth_min=0.3, depth_max=0.6):  # depth_min=0.3, depth_max=0.6
    # img = img_tensor.cpu().detach().numpy()

    if mode == 'rgb':
        img = img.transpose((1, 2, 0))
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if window is not None:
            cv2.imshow(window, img)
    elif mode == 'depth':
        img = img[0]
        img[img == 0] = 1000
        # print(np.max(img),np.min(img))
        # print("o",img)
        img = ((img - depth_min) * 255 / (depth_max - depth_min)).astype(np.uint8)

        # img = ((img - depth_min) * 255 / (depth_max - depth_min))
        # print("a",img)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        if window is not None:
            cv2.imshow(window, img)
    else:
        print("Not support mode!")
        return None
    return img



class HyperSim(Dataset):
    def __init__(self, csv_file, dataset_root):
        super(HyperSim, self).__init__()
        self.data = pd.read_csv(csv_file)
        self.dataset_root = dataset_root
        self.transform = transforms.Compose([
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        ])

    def __len__(self):
        return len(self.data)

    def random_crop_image_and_depth(self, image, depth, crop_size=(256, 256)):
        # Ensure the tensors have the same height and width
        assert image.size(1) == depth.size(1) and image.size(2) == depth.size(
            2), "Image and depth must have the same dimensions."

        # Get the maximum valid starting point (x, y) within the tensor bounds
        max_x = image.size(2) - crop_size[0]
        max_y = image.size(1) - crop_size[1]

        # Randomly select the upper-left corner of the crop
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)

        # Crop the image and depth tensors
        cropped_image = image[:, y:y + crop_size[1], x:x + crop_size[0]]
        cropped_depth = depth[:, y:y + crop_size[1], x:x + crop_size[0]]

        return cropped_image, cropped_depth

    def resize_color_and_depth_tensors(self, color_img, depth_img, size=(256, 256)):
        color_img = color_img.float()
        depth_img = depth_img.float()

        # Calculate the new size while maintaining the aspect ratio
        aspect_ratio = color_img.size(2) / color_img.size(1)
        new_width = int(size[1] * aspect_ratio)
        new_height = size[1]
        color_img = color_img.unsqueeze(0)
        depth_img = depth_img.unsqueeze(0)

        # Resize the color image
        color_resized = F.interpolate(color_img, size=(new_height, new_width), mode='bilinear', align_corners=False)

        # Resize the depth image
        depth_resized = F.interpolate(depth_img, size=(new_height, new_width), mode='bilinear', align_corners=False)

        # Crop the center of the resized images to (256, 256)

        color_resized = color_resized.squeeze(0)
        depth_resized = depth_resized.squeeze(0)

        # Perform center crop on the resized tensors
        center_x = (color_resized.size(2) - size[1]) // 2
        center_y = (color_resized.size(1) - size[0]) // 2

        color_cropped = color_resized[:, center_y:center_y + size[0], center_x:center_x + size[1]]
        depth_cropped = depth_resized[:, center_y:center_y + size[0], center_x:center_x + size[1]]

        return color_cropped, depth_cropped

    def resize_gt_depth_tensors(self, depth_img, size=(256, 256)):
        # color_img = color_img.float()
        depth_img = depth_img.float()

        # Calculate the new size while maintaining the aspect ratio
        aspect_ratio = depth_img.size(2) / depth_img.size(1)
        new_width = int(size[1] * aspect_ratio)
        new_height = size[1]
        # color_img = color_img.unsqueeze(0)
        depth_img = depth_img.unsqueeze(0)

        # Resize the color image
        # color_resized = F.interpolate(color_img, size=(new_height, new_width), mode='bilinear', align_corners=False)

        # Resize the depth image
        depth_resized = F.interpolate(depth_img, size=(new_height, new_width), mode='bilinear', align_corners=False)

        # Crop the center of the resized images to (256, 256)

        # color_resized = color_resized.squeeze(0)
        depth_resized = depth_resized.squeeze(0)

        # Perform center crop on the resized tensors
        center_x = (depth_resized.size(2) - size[1]) // 2
        center_y = (depth_resized.size(1) - size[0]) // 2
        #
        # color_cropped = color_resized[:, center_y:center_y + size[0], center_x:center_x + size[1]]
        depth_cropped = depth_resized[:, center_y:center_y + size[0], center_x:center_x + size[1]]

        return depth_cropped


    def __getitem__(self, idx):
        scene_name = self.data.iloc[idx, 0]
        camera_name = self.data.iloc[idx, 1]
        frame_id = self.data.iloc[idx, 2]

        hdf5_file = os.path.join(self.dataset_root, "scenes", scene_name, "images",
                                 f"scene_{camera_name}_geometry_hdf5", f"frame.{frame_id:04d}.depth_meters.hdf5")

        # Read depth image from HDF5 file
        with h5py.File(hdf5_file, 'r') as f:
            depth_image = f['dataset'][:]
        depth_image = self.transform(depth_image)
        depth_image = self.resize_gt_depth_tensors(depth_image, size=(228, 304))
        # You can process the data here as needed
        sample = {
            'depth_image': depth_image
        }

        return sample


import pandas as pd
from torch.utils.data import Dataset, DataLoader

import os
import h5py
import pandas as pd
from torch.utils.data import Dataset, DataLoader

import os
import h5py
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, csv_file, dataset_root, selected_csv_file, unselected_csv_file):
        self.data = pd.read_csv(csv_file)
        self.data = self.data[self.data['included_in_public_release']]
        self.dataset_root = dataset_root
        self.selected_csv_file = selected_csv_file
        self.unselected_csv_file = unselected_csv_file
        self.filtered_indices, self.unselected_indices = self._filter_samples()
        self._save_filtered_indices()

    def _filter_samples(self):
        filtered_indices = []
        unselected_indices = []
        for idx in range(len(self.data)):
            scene_name = self.data.iloc[idx, 0]
            camera_name = self.data.iloc[idx, 1]
            frame_id = self.data.iloc[idx, 2]
            hdf5_file = os.path.join(self.dataset_root, "scenes", scene_name, "images",
                                     f"scene_{camera_name}_geometry_hdf5", f"frame.{frame_id:04d}.depth_meters.hdf5")
            # filtered_data = self.data
            # filtered_data.to_csv(self.output_csv_file, index=False)
            # Read depth image from HDF5 file
            with h5py.File(hdf5_file, 'r') as f:
                depth_image = f['dataset'][:]

            # Check if depth image is within range [0, 10] and does not contain NaN values
            if np.any(np.isnan(depth_image)) or np.any((depth_image < 0) | (depth_image > 10)):
                print(scene_name, camera_name, frame_id)
                # Discard sample if depth image is not within range or contains NaN
                unselected_indices.append(idx)
            else:
                filtered_indices.append(idx)

        return filtered_indices, unselected_indices

    def _save_filtered_indices(self):
        selected_data = self.data.iloc[self.filtered_indices]
        unselected_data = self.data.iloc[self.unselected_indices]
        selected_data.to_csv(self.selected_csv_file, index=False)
        unselected_data.to_csv(self.unselected_csv_file, index=False)
        print("Saved filtered`")

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        idx = self.filtered_indices[idx]
        scene_name = self.data.iloc[idx, 0]
        camera_name = self.data.iloc[idx, 1]
        frame_id = self.data.iloc[idx, 2]
        exclude_reason = self.data.iloc[idx, 4]
        split_partition_name = self.data.iloc[idx, 5]

        hdf5_file = os.path.join(self.dataset_root, "scenes", scene_name, "images",
                                 f"scene_{camera_name}_geometry_hdf5", f"frame.{frame_id:04d}.depth_meters.hdf5")

        # Read depth image from HDF5 file
        with h5py.File(hdf5_file, 'r') as f:
            depth_image = f['dataset'][:]



        # You can process the data here as needed
        sample = {
            'depth_image': depth_image
        }

        return sample


if __name__ == '__main__':

    # Example usage:
    dataset_root = "/mnt/drive/Dataset/Hypersim"
    selected_csv_file = "selected_indices.csv"
    unselected_csv_file = "unselected_indices.csv"
    dataset = CustomDataset(os.path.join(dataset_root, "metadata_images_split_scene_v1.csv"), dataset_root,
                            selected_csv_file, unselected_csv_file)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for batch in dataloader:
        print(batch['depth_image'].shape)  # Assuming depth image is a numpy array
