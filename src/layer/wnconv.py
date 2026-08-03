
import torch 
from torch import nn
import torch.nn.functional as F 


class WNConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.weight_g = nn.Parameter(torch.empty(out_ch, 1, 1, 1))
        self.weight_v = nn.Parameter(torch.empty(out_ch, in_ch, 1, 1))
        self.bias = nn.Parameter(torch.empty(out_ch))
        self.reset_params()

    def weight(self):
        denom = self.weight_v.square().sum(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1e-12)
        return self.weight_v * (self.weight_g / denom)

    def forward(self, x):
        return F.conv2d(x, self.weight(), self.bias)
    
    def reset_params(self):
        with torch.no_grad():
            nn.init.kaiming_normal_(self.weight_v, mode='fan_in', nonlinearity='relu')
            # Initialize weight_g to 1.0 (variance-preserving) instead of
            # the L2 norm of weight_v (~1.4). The larger norm causes per-VQ
            # cascade amplification across residual stages, leading to VQ
            # loss explosion (>1e23) during early training.
            self.weight_g.fill_(1.0)
            self.bias.fill_(0)