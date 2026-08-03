# -*- encoding: utf-8 -*-
'''
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.

@File    :   vimeo90k_dataset.py
@Time    :   2026/03/18 15:25:58
@Author  :   XinChen
'''


import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
import numpy as np

from src.utils.transforms import ycbcr2rgb, rgb2ycbcr



class Vimeo90kDataset(Dataset):
    def __init__(self, root_dir, seq_len=7, patch_size=(256, 256), is_train=True, sequence="sequences", color_space="YCbCr", norm=False, image=False):
        """
        root_dir: Vimeo-90k dataset root directory, should contain a sequences folder
        seq_len: number of frames per video clip, usually 7
        patch_size: random crop patch size (H, W)
        is_train: enable random crop and flip in training mode
        sequence: dataset subdirectory, default "sequences"
        color_space: color space, "RGB" or "YCbCr", default "YCbCr"
        norm: whether to normalize images, True [0, 1] -> [-1, 1], False [0, 1] -> [0, 1]
        """
        self.color_space = color_space      # RGB or YCbCr
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.is_train = is_train
        self.root_dir = os.path.join(root_dir, sequence)
        self.video_list = self._make_dataset()
        self.to_tensor = transforms.ToTensor()
        self.norm = norm        # True [0, 1] -> [-1, 1]
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # [0, 1] -> [-1, 1]

        self.center_crop = transforms.CenterCrop(self.patch_size)
        self.image = image

    def set_frame_num(self, frame_num):
        self.seq_len = frame_num 
        # self.video_list = self._make_dataset()
    
    def get_frame_num(self):
        return self.seq_len 

    def _make_dataset(self):
        """Build a list of all video sequence paths, each is a subdirectory (containing 7 frames)."""
        video_list = []
        for root, _, files in os.walk(self.root_dir):
            if len(files) >= self.seq_len:
                video_list.append(root)
        return sorted(video_list)

    def _load_frames(self, video_path):
        """Load seq_len frames of the video, return List[PIL.Image]."""
        frames = []
        for i in range(1, self.seq_len + 1):
            img_path = os.path.join(video_path, f'im{i}.png')
            img = Image.open(img_path).convert('RGB')
            frames.append(img)
        return frames

    def _random_crop(self, images):
        """Apply the same random crop to a group of PIL images."""
        # TODO: add pad image
        w, h = images[0].size
        th, tw = self.patch_size
        pad_height = th - h 
        pad_width = tw - w 
        pad_height = max(0, pad_height)
        pad_width = max(0, pad_width)
        pad_size = ((pad_height // 2, pad_height - pad_height // 2),
                    (pad_width // 2, pad_width - pad_width // 2),
                    (0, 0), )
        # if w < tw or h < th:
        #     raise ValueError("Patch size larger than image size.")
        padded_height = pad_height + h 
        padded_width = pad_width + w
        x1 = random.randint(0, padded_width - tw)
        y1 = random.randint(0, padded_height - th)
        cropped = []
        # cropped = [img.crop((x1, y1, x1 + tw, y1 + th)) for img in images]
        for img in images:
            img = np.array(img).astype(np.uint8)
            img = np.pad(img, pad_size, mode='constant')
            img = img[y1:y1+self.patch_size[0], x1:x1+self.patch_size[1], :]
            cropped.append(Image.fromarray(img))
        return cropped

    def _augment(self, images):
        """Optional data augmentation: random horizontal flip."""
        if random.random() < 0.5:
            images = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in images]
        return images

    def __getitem__(self, index):
        video_path = self.video_list[index]
        images = self._load_frames(video_path)

        if self.is_train:
            # optional augmentation
            images = self._augment(images)
            images = self._random_crop(images)
            # if random.random() < 0.5:
            #     images = images[::-1]           # reverse sequence
                # np.random.shuffle(images)
        else:
            images = [self.center_crop(img) for img in images]

        # to Tensor and stack into [T, C, H, W]
        images = [self.to_tensor(img) for img in images]

        # @TODO: convert to yuv444 space
        if self.color_space.lower() == "ycbcr":
            images = [rgb2ycbcr(img) for img in images]

        if self.norm:
            images = [self.normalize(img) for img in images]        # [0, 1] -> [-1, 1]

        if self.image:
            return images[np.random.randint(0, self.seq_len)]

        frame_tensor = torch.stack(images, dim=0)

        return frame_tensor  # shape: [T, C, H, W]

    def __len__(self):
        return len(self.video_list)
    
    def set_patch_size(self, patch_size):
        # (H, W)
        self.patch_size = patch_size 
    
    def get_patch_size(self):
        return self.patch_size # (H, W)
    
    
    