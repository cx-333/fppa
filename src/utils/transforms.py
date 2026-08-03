import numpy as np
import scipy.ndimage
import torch
import torch.nn.functional as F


YCBCR_WEIGHTS = {
    # Spec: (K_r, K_g, K_b) with K_g = 1 - K_r - K_b
    "ITU-R_BT.709": (0.2126, 0.7152, 0.0722)
}


def to_01(img: torch.Tensor) -> torch.Tensor:
    """Convert image tensor from [-1, 1] to [0, 1]."""
    return (img + 1.0) / 2.0


def to_m11(img: torch.Tensor) -> torch.Tensor:
    """Convert image tensor from [0, 1] to [-1, 1]."""
    return img * 2.0 - 1.0


def assert_value_range(img: torch.Tensor, low: float, high: float, name: str = 'image', eps: float = 1e-4) -> None:
    """Cheap range assertion for debugging value-range mistakes."""
    if not torch.is_tensor(img):
        raise TypeError(f'{name} must be torch.Tensor, got {type(img)}')
    if img.numel() == 0:
        return
    min_val = float(img.detach().amin())
    max_val = float(img.detach().amax())
    if min_val < low - eps or max_val > high + eps:
        raise ValueError(f'{name} range [{min_val:.6f}, {max_val:.6f}] is outside [{low}, {high}]')


def rgb2ycbcr_m11(rgb: torch.Tensor) -> torch.Tensor:
    """RGB [-1, 1] -> YCbCr [-1, 1]."""
    return to_m11(rgb2ycbcr(to_01(rgb)))


def ycbcr2rgb_m11(ycbcr: torch.Tensor) -> torch.Tensor:
    """YCbCr [-1, 1] -> RGB [-1, 1]."""
    return to_m11(ycbcr2rgb(to_01(ycbcr)))


def ycbcr420_to_444_np(y, uv, order=0, separate=False):
    '''
    y is 1xhxw Y float numpy array
    uv is 2x(h/2)x(w/2) UV float numpy array
    order: 0 nearest neighbor (default), 1: binear
    return value is 3xhxw YCbCr float numpy array
    '''
    uv = scipy.ndimage.zoom(uv, (1, 2, 2), order=order)
    if separate:
        return y, uv
    yuv = np.concatenate((y, uv), axis=0)
    return yuv


def rgb2ycbcr(rgb, is_bgr=False):
    if is_bgr:
        b, g, r = rgb.chunk(3, -3)
    else:
        r, g, b = rgb.chunk(3, -3)
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    y = Kr * r + Kg * g + Kb * b
    cb = 0.5 * (b - y) / (1 - Kb) + 0.5
    cr = 0.5 * (r - y) / (1 - Kr) + 0.5
    ycbcr = torch.cat((y, cb, cr), dim=-3)
    ycbcr = torch.clamp(ycbcr, 0., 1.)      # 【0， 1】
    return ycbcr


def ycbcr2rgb(ycbcr, is_bgr=False, clamp=True):
    y, cb, cr = ycbcr.chunk(3, -3)
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    r = y + (2 - 2 * Kr) * (cr - 0.5)
    b = y + (2 - 2 * Kb) * (cb - 0.5)
    g = (y - Kr * r - Kb * b) / Kg
    if is_bgr:
        rgb = torch.cat((b, g, r), dim=-3)
    else:
        rgb = torch.cat((r, g, b), dim=-3)
    if clamp:
        rgb = torch.clamp(rgb, 0., 1.)
    return rgb


def yuv_444_to_420(yuv):
    # model output to yuv420
    def _downsample(tensor):
        return F.avg_pool2d(tensor, kernel_size=2, stride=2)

    y = yuv[:, :1, :, :]
    uv = yuv[:, 1:, :, :]
    # 
    return y, _downsample(uv)
