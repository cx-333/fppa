# -*- encoding: utf-8 -*-
'''
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.

@File    :   imagenet_dataset.py
@Time    :   2026/03/18 15:42:30
@Author  :   XinChen
'''

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
import numpy as np
from src.utils.transforms import ycbcr2rgb, rgb2ycbcr
from src.utils.utils import get_padding_size, replicated_pad


class ImageNetDataset(Dataset):
    def __init__(self, root_dir, patch_size=(256, 256), is_train=True, color_space="ycbcr", norm=False):
        self.patch_size = patch_size        # (H, W)
        self.is_train = is_train
        self.root_dir = root_dir
        self.color_space = color_space      # or YCbCr
        self.image_list = self._make_dataset()
        self.norm = norm        # True [0, 1] -> [-1, 1]

        # if self.is_train:
        #     self.transform = transforms.Compose([
        #         # transforms.RandomCrop(patch_size), # crop in PIL stage to save memory and speed up
        #         # transforms.RandomHorizontalFlip(p=0.5),
        #         transforms.ToTensor()              # to Tensor and normalize to [0, 1]
        #     ])
        # else:
        #     self.transform = transforms.Compose([
        #         # transforms.CenterCrop(patch_size), # center crop for deterministic validation
        #         transforms.ToTensor()
        #     ])
        self.transform = transforms.ToTensor()
        
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # [0, 1] -> [-1, 1]


    def _make_dataset(self):
        # tuple of valid extensions, endswith performs parallel matching internally
        valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP")

        with os.scandir(self.root_dir) as entries:
            return [e.path for e in entries if e.is_file() and e.name.endswith(valid_extensions)]

    def __len__(self):
        return len(self.image_list)

    def _augment(self, image):
        if random.random() < 0.5:
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        return image

    def __getitem__(self, index):
        img_path = self.image_list[index]
        image = Image.open(img_path).convert("RGB")

        # train & val 
        if self.is_train:
            image = self._augment(image)
            x, y, pad_size = self.pad_image(image)
            
            image = self.transform(image)
            
            image = torch.nn.functional.pad(image, pad_size, mode="constant", value = 0)
            image = image[:, y:y+self.patch_size[0], x:x+self.patch_size[1]]
        
        else:
            image = self.transform(image)
            p_b, p_r = get_padding_size(*image.size()[-2:], p=64)
            image = replicated_pad(image, p_b, p_r)

        if self.color_space.lower() == "ycbcr":
            # @TODO: convert to yuv444 space
            image = rgb2ycbcr(image)
        if self.norm:
            image = self.normalize(image)   # [0, 1] -> [-1, 1]

        return image

    def pad_image(self, image):
        width, height = image.size 
        pad_height = self.patch_size[0] - height
        pad_width = self.patch_size[1] - width 
        pad_height = max(0, pad_height)
        pad_width = max(0, pad_width)
        # (l, r, t, b)
        pad_size = (pad_width // 2, pad_width - pad_width // 2, pad_height // 2, pad_height - pad_height // 2)
        padded_height = height + pad_height 
        padded_width = width + pad_width 
        y = random.randint(0, padded_height - self.patch_size[0])
        x = random.randint(0, padded_width - self.patch_size[1])
        return x, y, pad_size 


    def set_patch_size(self, patch_size):
        # (H, W)
        self.patch_size = patch_size 
        
    def get_patch_size(self):
        # (H, W)
        return self.patch_size 
    