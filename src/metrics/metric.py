# -*- encoding: utf-8 -*-
"""

@File    :   metric.py
@Time    :   2026/07/05 10:42:51
@Author  :   XinChen
"""

import torch
import torch.nn.functional as F

try:
    from pytorch_msssim import ssim as _msssim_ssim
except ImportError:
    _msssim_ssim = None


def _ssim_fallback(x, y, data_range=1.0, size_average=True):
    """Simple SSIM implementation as fallback when pytorch_msssim is unavailable.

    Uses the standard SSIM formula with fixed Gaussian window (11x11, sigma=1.5).
    """
    from math import exp
    import numpy as np

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    # Create 1D Gaussian kernel
    def gaussian_kernel(size, sigma):
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g

    kernel_1d = gaussian_kernel(11, 1.5).to(x.device)
    window = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)  # [11, 11]
    window = window.unsqueeze(0).unsqueeze(0)  # [1, 1, 11, 11]
    window = window.expand(x.size(1), 1, 11, 11)  # [C, 1, 11, 11]

    groups = x.size(1)

    mu_x = F.conv2d(x, window, groups=groups, padding=5)
    mu_y = F.conv2d(y, window, groups=groups, padding=5)

    sigma_x_sq = F.conv2d(x * x, window, groups=groups, padding=5) - mu_x ** 2
    sigma_y_sq = F.conv2d(y * y, window, groups=groups, padding=5) - mu_y ** 2
    sigma_xy = F.conv2d(x * y, window, groups=groups, padding=5) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x_sq + sigma_y_sq + C2))

    if size_average:
        return ssim_map.mean()
    return ssim_map.mean(dim=[1, 2, 3])


def compute_ssim(x, y, data_range=1.0, size_average=True):
    """Compute SSIM, using pytorch_msssim if available, else fallback."""
    if _msssim_ssim is not None:
        return _msssim_ssim(x, y, data_range=data_range, size_average=size_average)
    return _ssim_fallback(x, y, data_range=data_range, size_average=size_average)


@torch.no_grad()
def calculate_metrics(x_hat, target):
    """Calculate PSNR and SSIM (returns scalar values)."""

    x_hat = torch.clamp(x_hat, 0., 1.)
    target = torch.clamp(target, 0., 1.)

    mse = F.mse_loss(x_hat, target, reduction='mean')
    psnr = (10 * torch.log10(1.0 / mse)).item() if mse.item() > 0 else 100.0
    ssim_val = compute_ssim(x_hat, target, data_range=1.0, size_average=True).item()

    return psnr, ssim_val
