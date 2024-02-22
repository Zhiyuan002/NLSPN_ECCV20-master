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



class ToFDataset(Dataset):
    def __init__(self, data_split):
        super(ToFDataset, self).__init__()
        self.transform = transforms.Compose([
        # transforms.Resize((256, 256)),  # Resize the image to a desired size
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        ])
        self.mode = data_split
        # with open("matterport3d_train_toy.json", "r", encoding="utf-8") as f:
        #     self.depth_file_paths_M3d = json.load(f)

        with open("DydToF_image_all.json", "r", encoding="utf-8") as f:
            self.depth_file_paths_ToF = json.load(f)
        if self.mode == 'train':
            self.depth_file_paths_ToF = self.depth_file_paths_ToF[0:45205]
        if self.mode == 'val':
            self.depth_file_paths_ToF = self.depth_file_paths_ToF[45205:45205+2379]
        # print(self.depth_file_paths_M3d)
        # print(len(self.depth_file_paths_ToF),len(self.depth_file_paths_M3d))
        # self.depth_file_paths = self.image_file_paths
        # print(self.depth_file_paths)

    def __len__(self):
        return len(self.depth_file_paths_ToF)

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

        depth_path_ToF = self.depth_file_paths_ToF[idx][0]
        # print(depth_path_ToF)
        image_path_ToF = depth_path_ToF.replace("DepthMap", "ColorImage").replace(".npy", ".jpeg")
        # image_path = image_path.replace("iepth", "color")
        # print(image_path, depth_path)
        image_ToF = Image.open(image_path_ToF)
        depth_ToF = np.load(depth_path_ToF)
        # print(image)
        # print(depth.max(), depth.min())
        image_ToF = self.transform(image_ToF)
        depth_ToF = self.transform(depth_ToF)
        # Assuming color_img and depth_img are your input tensors
        image_ToF, depth_ToF = self.resize_color_and_depth_tensors(image_ToF, depth_ToF, size=(228, 304))

        # print(depth_ToF.min(), depth_ToF.max(),depth_ToF.mean(), depth_ToF.std())
        # print(depth_ToF.min(),depth_ToF.max(),depth_ToF.mean(),depth_ToF.std())

        # print(mask)
        sample = {
            'color_image_ToF': image_ToF,
            'depth_image_ToF': depth_ToF,
        }
        # print(sample['color_image'])
        # print(min(gt_M3d))
        # print(torch.max(gt_M3d), torch.max(depth_M3d))

        # visualize_tensor(gt_M3d, 'depth', window='rgb1',depth_min=0, depth_max= 20000)

        # cv2.waitKey(1000)
        return sample
