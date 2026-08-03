
import torch
from torch import nn
import torch.nn.functional as F


class FastConv2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, **kwargs)

class FastEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int | None = None, max_norm: float | None = None, norm_type: float = 2, scale_grad_by_freq: bool = False, sparse: bool = False, _weight: torch.Tensor | None = None, _freeze: bool = False, device=None, dtype=None) -> None:
        super().__init__(num_embeddings, embedding_dim, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse, _weight, _freeze, device, dtype)

def conv(in_ch: int, out_ch: int, kernel_size: int = 5, stride: int = 1, bias: bool = True):
    return FastConv2d(in_ch, out_ch, kernel_size, stride=stride, padding=kernel_size // 2, bias=bias)


class ReLUChunkAdd(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        x = F.relu(x, inplace=False)
        x1, x2 = x.chunk(2, 1)
        return x1 + x2


class DepthConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, shortcut: bool = False, force_adaptor: bool = False):
        super().__init__()
        self.adaptor = FastConv2d(in_ch, out_ch, 1) if (in_ch != out_ch or force_adaptor) else None
        self.shortcut = shortcut
        self.dc = nn.Sequential(
            FastConv2d(out_ch, out_ch, 1),
            nn.ReLU(inplace=False),
            FastConv2d(out_ch, out_ch, 3, padding=1, groups=out_ch),
            FastConv2d(out_ch, out_ch, 1),
        )
        self.ffn = nn.Sequential(
            FastConv2d(out_ch, out_ch * 4, 1),
            ReLUChunkAdd(),
            FastConv2d(out_ch * 2, out_ch, 1),
        )

    def forward(self, x):
        if self.adaptor is not None:
            x = self.adaptor(x)
        out = self.dc(x)
        out = out + x
        out = out + self.ffn(out)
        if self.shortcut:
            out = out + x
        return out


class SubpelConv2x(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(FastConv2d(in_ch, out_ch * 4, 1), nn.PixelShuffle(2))

    def forward(self, x):
        return self.conv(x)


class ResidualBlockWithStride2(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = FastConv2d(in_ch, out_ch, 2, stride=2)
        self.conv = DepthConvBlock(out_ch, out_ch, shortcut=True)

    def forward(self, x):
        return self.conv(self.down(x))



class ResidualBlockUpsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = SubpelConv2x(in_ch, out_ch)
        self.conv = DepthConvBlock(out_ch, out_ch, shortcut=True)

    def forward(self, x):
        return self.conv(self.up(x))
