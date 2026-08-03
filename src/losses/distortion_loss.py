from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from pytorch_msssim import MS_SSIM

from src.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    def __init__(self,
                 loss_weight: float,
                 normalize_img: bool=True,
                 mse_scale: str='0_1'):
        """_summary_

        Args:
            loss_weight (float): _description_
            normalize_img (bool, optional):
            mse_scale (str, optional): Scale of pixel values for MSE loss
        """
        super().__init__()
        assert normalize_img
        assert mse_scale in ['0_255', '0_1'], f'mse_scale should be "0_255" or "0_1", but {mse_scale} ({type(mse_scale)})'
        self.lamb_mse = loss_weight
        self.mse = nn.MSELoss(reduction='none')
        normalize_func_dict = {'0_255': self.img_range_to_255, '0_1': self.img_range_to_01}
        self.normalize_func = normalize_func_dict[mse_scale]

    @staticmethod
    def img_range_to_255(img: Tensor) -> Tensor:
        img = (img + 1.) / 2. # [-1, 1] -> [0, 1]
        return img * 255. # [0, 1] -> [0, 255]

    @staticmethod
    def img_range_to_01(img: Tensor) -> Tensor:
        return (img + 1.) / 2. # [-1, 1] -> [0, 1]

    def forward(self, real_images: Tensor, fake_images: Tensor, **kwargs):
        if self.normalize_func:
            real_images = self.normalize_func(real_images)
            fake_images = self.normalize_func(fake_images)
        mse = self.mse(real_images, fake_images).mean(dim=[1, 2, 3])
        return self.lamb_mse * mse          # (B, )

@LOSS_REGISTRY.register()
class MultiMSELoss(nn.Module):
    def __init__(self, loss_weights: List[float], device: str='cuda'):
        super().__init__()
        self.loss_weights = torch.tensor(loss_weights, dtype=torch.float, device=device)
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, real_images: Tensor, fake_images: Tensor, qp: Tensor, **kwargs):
        real_images = MSELoss.img_range_to_01(real_images)
        fake_images = MSELoss.img_range_to_01(fake_images)
        loss_weights = self.loss_weights[qp]
        mse_loss = self.mse(real_images, fake_images).mean(dim=[1, 2, 3])
        return loss_weights * mse_loss      # (B, )


@LOSS_REGISTRY.register()
class DCVCRTMSELoss(nn.Module):
    def __init__(self, loss_weights: List[float], device: str='cuda'):
        super().__init__()
        self.lamb_mse = torch.tensor(loss_weights, dtype=torch.float, device=device)
        self.k = 0.8
        self.mse = nn.MSELoss(reduction='none')

    @staticmethod
    def _to_01(img: Tensor) -> Tensor:
        return (img + 1.0) / 2.0  # [-1, 1] -> [0, 1]

    def forward(self, real_images: Tensor, fake_images: Tensor, real_ycbcr_images: Tensor, fake_ycbcr_images: Tensor, qp: Tensor, **kwargs):
        # real_images, fake_images: RGB
        # real_ycbcr_images, fake_ycbcr_images: YUV
        # NOTE: inputs from run_comp_model are in [-1, 1]; normalize to [0, 1] for consistent MSE
        real_images = self._to_01(real_images)
        fake_images = self._to_01(fake_images)
        real_ycbcr_images = self._to_01(real_ycbcr_images)
        fake_ycbcr_images = self._to_01(fake_ycbcr_images)
        yuv_mse = self.mse(real_ycbcr_images, fake_ycbcr_images).mean(dim=[1, 2, 3])
        rgb_mse = self.mse(real_images, fake_images).mean(dim=[1, 2, 3])    
        return self.lamb_mse[qp] * (self.k * yuv_mse + (1 - self.k) * rgb_mse)          # (B, )


@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    def __init__(self, loss_weight: float):
        super().__init__()
        self.lamb_l1 = loss_weight
        self.l1_loss = nn.L1Loss(reduction='none')

    @staticmethod
    def _to_01(img: Tensor) -> Tensor:
        return (img + 1.0) / 2.0  # [-1, 1] -> [0, 1]

    def forward(self, real_images, fake_images, **kwargs):
        # NOTE: inputs from run_comp_model are in [-1, 1]; normalize to [0, 1] for consistent L1
        real_images = self._to_01(real_images)
        fake_images = self._to_01(fake_images)
        l1 = self.l1_loss(real_images, fake_images).mean(dim=[1, 2, 3])
        return self.lamb_l1 * l1          # (B, )   


@LOSS_REGISTRY.register()
class MSSSIMLoss(nn.Module):
    def __init__(self, loss_weight: float):
        super().__init__()
        self.lamb_msssim = loss_weight
        self.ms_ssim_loss = MS_SSIM(data_range=1, size_average=False, channel=3)

    @staticmethod
    def _to_01(img: Tensor) -> Tensor:
        return (img + 1.0) / 2.0  # [-1, 1] -> [0, 1]

    def forward(self, real_images, fake_images, **kwargs):
        # NOTE: inputs from run_comp_model are in [-1, 1]; MS_SSIM(data_range=1) expects [0, 1]
        real_images = self._to_01(real_images)
        fake_images = self._to_01(fake_images)
        msssim = self.ms_ssim_loss(real_images, fake_images)
        return self.lamb_msssim * (1 - msssim)          # (B, )

