

from src import loralib 
import torch 
from torch import nn 
from .layers import WSiLU, WSiLUChunkAdd


class SubpelConv2x(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding=0, r=24, conv_lora_expert_num=12):
        super().__init__()
        self.conv = nn.Sequential(
            loralib.MoEConv2d(in_ch, out_ch * 4, kernel_size=kernel_size, padding=padding, r=r, conv_lora_expert_num=conv_lora_expert_num),
            nn.PixelShuffle(2)
            )
        # self.up = nn.PixelShuffle(2)
        self.padding = padding

        self.proxy = None

    def forward(self, x, cond, to_cat=None, cat_at_front=True):
        return self.forward_torch(x, cond, to_cat, cat_at_front)

    def forward_torch(self, x, cond, to_cat=None, cat_at_front=True):
        out, loss = self.conv[0](x, cond)
        out = self.conv[1](out)
        if to_cat is None:
            return out, loss
        if cat_at_front:
            return torch.cat((to_cat, out), dim=1), loss
        return torch.cat((out, to_cat), dim=1), loss


class DepthConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, shortcut=False, force_adaptor=False, r=24, conv_lora_expert_num=12):
        super().__init__()
        self.adaptor = None
        if in_ch != out_ch or force_adaptor:
            self.adaptor = loralib.MoEConv2d(in_ch, out_ch, 1, r=r, conv_lora_expert_num=conv_lora_expert_num)
        self.shortcut = shortcut
        self.dc = nn.Sequential(
            loralib.MoEConv2d(out_ch, out_ch, 1, r=r, conv_lora_expert_num=conv_lora_expert_num),
            WSiLU(),
            loralib.DepthConvLoRA(nn.Conv2d, out_ch, out_ch, 3, padding=1, groups=out_ch, r=5),
            loralib.MoEConv2d(out_ch, out_ch, 1, r=r, conv_lora_expert_num=conv_lora_expert_num),
        )
        self.ffn = nn.Sequential(
            loralib.MoEConv2d(out_ch, out_ch * 4, 1, r=r, conv_lora_expert_num=conv_lora_expert_num),
            WSiLUChunkAdd(),
            loralib.MoEConv2d(out_ch * 2, out_ch, 1, r=r, conv_lora_expert_num=conv_lora_expert_num),
        )

        self.proxy = None

    def forward(self, x, cond, quant_step=None, to_cat=None, cat_at_front=True):
        return self.forward_torch(x, cond, quant_step, to_cat, cat_at_front)

    def forward_torch(self, x, cond, quant_step=None, to_cat=None, cat_at_front=True):
        total_loss = 0.
        if self.adaptor is not None:
            x, loss = self.adaptor(x, cond)
            total_loss = total_loss + loss 
        # out = self.dc(x) + x
        out = x 
        for i in range(len(self.dc)):
            if isinstance(self.dc[i], (loralib.MoEConv2d)):
                out, loss = self.dc[i](out, cond)
                total_loss = total_loss + loss 
            else:
                out = self.dc[i](out)
        out = out + x     
        # out = self.ffn(out) + out
        out1, loss1 = self.ffn[0](out, cond)
        out1 = self.ffn[1](out1)
        out1, loss2 = self.ffn[2](out1, cond)
        total_loss = total_loss + loss1 + loss2 
        out = out1 + out 
        
        if self.shortcut:
            out = out + x
        if quant_step is not None:
            out = out * quant_step
        if to_cat is not None:
            if cat_at_front:
                out = torch.cat((to_cat, out), dim=1)
            else:
                out = torch.cat((out, to_cat), dim=1)
        return out, total_loss 
    
    
class ResidualBlockWithStride2(nn.Module):
    def __init__(self, in_ch, out_ch, r=24, conv_lora_expert_num=12):
        super().__init__()
        self.down = loralib.MoEConv2d(in_ch, out_ch, 2, stride=2, r=r, conv_lora_expert_num=conv_lora_expert_num)
        self.conv = DepthConvBlock(out_ch, out_ch, shortcut=True, r=r, conv_lora_expert_num=conv_lora_expert_num)

    def forward(self, x, cond):
        x, loss1 = self.down(x, cond)
        out, loss2 = self.conv(x, cond)
        return out, loss1+loss2 
    

class ResidualBlockUpsample(nn.Module):
    def __init__(self, in_ch, out_ch, r=24, conv_lora_expert_num=12):
        super().__init__()
        self.up = SubpelConv2x(in_ch, out_ch, 1, r=r, conv_lora_expert_num=conv_lora_expert_num)
        self.conv = DepthConvBlock(out_ch, out_ch, shortcut=True, r=r, conv_lora_expert_num=conv_lora_expert_num)

    def forward(self, x, cond):
        out, loss1 = self.up(x, cond)
        out, loss2 = self.conv(out, cond)
        return out, loss1+loss2 
    
    