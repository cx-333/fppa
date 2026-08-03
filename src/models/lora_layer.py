
from src import loralib 
import torch 
from torch import nn 
from .layers import WSiLU, WSiLUChunkAdd



class SubpelConv2x(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding=0):
        super().__init__()
        r = int(min(in_ch, out_ch) * 0.2)
        self.conv = nn.Sequential(
            loralib.Conv2d(in_ch, out_ch * 4, kernel_size=kernel_size, padding=padding, r=r),
            nn.PixelShuffle(2),
        )
        self.padding = padding

        self.proxy = None

    def forward(self, x, to_cat=None, cat_at_front=True):
        return self.forward_torch(x, to_cat, cat_at_front)

    def forward_torch(self, x, to_cat=None, cat_at_front=True):
        out = self.conv(x)
        if to_cat is None:
            return out
        if cat_at_front:
            return torch.cat((to_cat, out), dim=1)
        return torch.cat((out, to_cat), dim=1)


class DepthConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, shortcut=False, force_adaptor=False):
        super().__init__()
        self.adaptor = None
        r = int(min(in_ch, out_ch) * 0.2)
        if in_ch != out_ch or force_adaptor:
            # self.adaptor = nn.Conv2d(in_ch, out_ch, 1)
            self.adaptor = loralib.Conv2d(in_ch, out_ch, 1, r=int(min(in_ch, out_ch) * 0.1))
        self.shortcut = shortcut
        self.dc = nn.Sequential(
            loralib.Conv2d(out_ch, out_ch, 1, r=r),
            WSiLU(),
            loralib.DepthConvLoRA(nn.Conv2d, out_ch, out_ch, 3, padding=1, groups=out_ch, r=5),
            loralib.Conv2d(out_ch, out_ch, 1, r=r),
        )
        self.ffn = nn.Sequential(
            loralib.Conv2d(out_ch, out_ch * 4, 1, r=r),
            WSiLUChunkAdd(),
            loralib.Conv2d(out_ch * 2, out_ch, 1, r=r),
        )

        self.proxy = None

    def forward(self, x, quant_step=None, to_cat=None, cat_at_front=True):
        return self.forward_torch(x, quant_step, to_cat, cat_at_front)

    def forward_torch(self, x, quant_step=None, to_cat=None, cat_at_front=True):
        if self.adaptor is not None:
            x = self.adaptor(x)
        out = self.dc(x) + x
        out = self.ffn(out) + out
        if self.shortcut:
            out = out + x
        if quant_step is not None:
            out = out * quant_step
        if to_cat is not None:
            if cat_at_front:
                out = torch.cat((to_cat, out), dim=1)
            else:
                out = torch.cat((out, to_cat), dim=1)
        return out
    
    
class ResidualBlockWithStride2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # self.down = nn.Conv2d(in_ch, out_ch, 2, stride=2)
        self.down = loralib.Conv2d(in_ch, out_ch, 2, stride=2, r=int(min(in_ch, out_ch) * 0.2))
        self.conv = DepthConvBlock(out_ch, out_ch, shortcut=True)

    def forward(self, x):
        x = self.down(x)
        out = self.conv(x)
        return out
    

class ResidualBlockUpsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = SubpelConv2x(in_ch, out_ch, 1)
        self.conv = DepthConvBlock(out_ch, out_ch, shortcut=True)

    def forward(self, x):
        out = self.up(x)
        out = self.conv(out)
        return out
    
    