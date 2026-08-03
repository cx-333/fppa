# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torch import nn
import torch.nn.functional as F


class WSiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sigmoid(4.0 * x) * x


class WSiLUChunkAdd(nn.Module):
    def __init__(self):
        super().__init__()
        self.silu = WSiLU()

    def forward(self, x):
        x1, x2 = self.silu(x).chunk(2, 1)
        return x1 + x2


class SubpelConv2x(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=kernel_size, padding=padding),
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
        if in_ch != out_ch or force_adaptor:
            self.adaptor = nn.Conv2d(in_ch, out_ch, 1)
        self.shortcut = shortcut
        self.dc = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1),
            WSiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch),
            nn.Conv2d(out_ch, out_ch, 1),
        )
        self.ffn = nn.Sequential(
            nn.Conv2d(out_ch, out_ch * 4, 1),
            WSiLUChunkAdd(),
            nn.Conv2d(out_ch * 2, out_ch, 1),
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
        self.down = nn.Conv2d(in_ch, out_ch, 2, stride=2)
        self.conv = DepthConvBlock(out_ch, out_ch, shortcut=True)

    def forward(self, x):
        x = self.down(x)
        out = self.conv(x)
        return out

class DepthConv(nn.Module):
    def __init__(self, in_ch, out_ch, depth_kernel=3, stride=1, slope=0.01, inplace=False):
        super().__init__()
        dw_ch = in_ch * 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, dw_ch, 1, stride=stride),
            nn.LeakyReLU(negative_slope=slope, inplace=inplace),
        )
        self.depth_conv = nn.Conv2d(dw_ch, dw_ch, depth_kernel, padding=depth_kernel // 2,
                                    groups=dw_ch)
        self.conv2 = nn.Conv2d(dw_ch, out_ch, 1)

        self.adaptor = None
        if stride != 1:
            assert stride == 2
            self.adaptor = nn.Conv2d(in_ch, out_ch, 2, stride=2)
        elif in_ch != out_ch:
            self.adaptor = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        identity = x
        if self.adaptor is not None:
            identity = self.adaptor(identity)

        out = self.conv1(x)
        out = self.depth_conv(out)
        out = self.conv2(out)

        return out + identity
    
class ConvFFN2(nn.Module):
    def __init__(self, in_ch, slope=0.1, inplace=False):
        super().__init__()
        expansion_factor = 2
        slope = 0.1
        internal_ch = in_ch * expansion_factor
        self.conv = nn.Conv2d(in_ch, internal_ch * 2, 1)
        self.conv_out = nn.Conv2d(internal_ch, in_ch, 1)
        self.relu = nn.LeakyReLU(negative_slope=slope, inplace=inplace)

    def forward(self, x):
        identity = x
        x1, x2 = self.conv(x).chunk(2, 1)
        out = x1 * self.relu(x2)
        return identity + self.conv_out(out)
    
class ResidualBlockUpsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = SubpelConv2x(in_ch, out_ch, 1)
        self.conv = DepthConvBlock(out_ch, out_ch, shortcut=True)

    def forward(self, x):
        out = self.up(x)
        out = self.conv(out)
        return out

class DepthConvBlock2(nn.Module):
    def __init__(self, in_ch, out_ch, depth_kernel=3, stride=1,
                 slope_depth_conv=0.01, slope_ffn=0.1, inplace=False):
        super().__init__()
        self.block = nn.Sequential(
            DepthConv(in_ch, out_ch, depth_kernel, stride, slope=slope_depth_conv, inplace=inplace),
            ConvFFN2(out_ch, slope=slope_ffn, inplace=inplace),
        )

    def forward(self, x):
        return self.block(x)
    
    
class UNet2(nn.Module):
    def __init__(self, in_ch=192, out_ch=192, inplace=False):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = DepthConvBlock2(in_ch, in_ch//2, inplace=inplace)
        self.conv2 = DepthConvBlock2(in_ch//2, in_ch, inplace=inplace)
        self.conv3 = DepthConvBlock2(in_ch, in_ch, inplace=inplace)

        self.context_refine = nn.Sequential(
            DepthConvBlock2(in_ch, in_ch, inplace=inplace),
            DepthConvBlock2(in_ch, in_ch, inplace=inplace),
            DepthConvBlock2(in_ch, in_ch, inplace=inplace),
            DepthConvBlock2(in_ch, in_ch, inplace=inplace),
        )

        self.up3 = subpel_conv1x1(in_ch, in_ch//2, 2)
        self.up_conv3 = DepthConvBlock2(in_ch + in_ch//2, in_ch//2, inplace=inplace)

        self.up2 = subpel_conv1x1(in_ch//2, in_ch//2, 2)
        self.up_conv2 = DepthConvBlock2(in_ch//2 + in_ch//2, out_ch, inplace=inplace)

    def forward(self, x):
        # encoding path
        x1 = self.conv1(x)
        x2 = self.max_pool(x1)

        x2 = self.conv2(x2)
        x3 = self.max_pool(x2)

        x3 = self.conv3(x3)
        x3 = self.context_refine(x3)

        # decoding + concat path
        d3 = self.up3(x3)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up_conv2(d2)
        return d2
    
#=======================================================================================================================#
def subpel_conv1x1(in_ch, out_ch, r=1):
    """1x1 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r ** 2, kernel_size=1, padding=0), nn.PixelShuffle(r)
    )
    
def conv3x3(in_ch, out_ch, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)

    
def build_index_dec(scales, scale_min, scale_max, log_scale_min, log_step_recip, skip_thres=None):
    skip_cond = None
    scales = scales.clamp_(scale_min, scale_max)
    indexes = (torch.log(scales) - log_scale_min) * log_step_recip
    indexes = indexes.to(dtype=torch.uint8)
    if skip_thres is not None:
        skip_cond = scales > skip_thres
    return indexes, skip_cond


def build_index_enc(symbols, scales, scale_min, scale_max, log_scale_min,
                    log_step_recip, skip_thres=None):
    scales = scales.clamp_(scale_min, scale_max)
    indexes = (torch.log(scales) - log_scale_min) * log_step_recip
    indexes = indexes.to(dtype=torch.uint8)
    symbols = symbols.to(dtype=torch.int16)
    out = (symbols << 8) + indexes
    out = out.to(dtype=torch.int16)
    if skip_thres is not None:
        skip_cond = scales > skip_thres
        out = out[skip_cond]
    return out


def process_with_mask(y, scales, means, mask, force_zero_thres):
    scales_hat = scales * mask
    means_hat = means * mask
    y_res = (y - means_hat) * mask
    y_q =  torch.round(y_res)           # 推理时使用，梯度会断
    if force_zero_thres is not None:
        cond = scales_hat > force_zero_thres
        y_q = y_q * cond
    y_q = torch.clamp(y_q, -128., 127.)
    y_hat = y_q + means_hat
    return y_res, y_q, y_hat, scales_hat


def combine_for_reading_2x(x, mask, inplace=False):
    x = x * mask
    x0, x1 = x.chunk(2, 1)
    return x0 + x1

def restore_y_2x(y, means, mask):
    return (torch.cat((y, y), dim=1) + means) * mask

def restore_y_2x_with_cat_after(y, means, mask, to_cat):
    out = (torch.cat((y, y), dim=1) + means) * mask
    return out, torch.cat((out, to_cat), dim=1)

def add_and_multiply(y_hat_0, y_hat_1, q_dec):
    y_hat = y_hat_0 + y_hat_1
    y_hat = y_hat * q_dec
    return y_hat

def replicate_pad(x, pad_b, pad_r):
    if pad_b == 0 and pad_r == 0:
        return x
    return F.pad(x, (0, pad_r, 0, pad_b), mode="replicate")

def restore_y_4x(y, means, mask):
    return (torch.cat((y, y, y, y), dim=1) + means) * mask


def clamp_reciprocal_with_quant(q_dec, y, min_val=0.5):
    q_dec = torch.clamp_min(q_dec, min_val)
    q_enc = torch.reciprocal(q_dec)
    y = y * q_enc
    return q_dec, y

#++++++++++++++++++++++++++++++++++++ Spatial Feature Transfrom +++++++++++++++++++++++++++++++++++++#
class SFTLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(in_ch, in_ch, 1)
        self.SFT_scale_conv1 = nn.Conv2d(in_ch, out_ch, 1)
        self.SFT_shift_conv0 = nn.Conv2d(in_ch, in_ch, 1)
        self.SFT_shift_conv1 = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: condition (qp -> maps)
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.1, inplace=True))   # gamma
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x[1]), 0.1, inplace=True))   # beta
        return x[0] * scale + shift


class ResBlock_SFT(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResBlock_SFT, self).__init__()
        self.sft0 = SFTLayer(in_ch, out_ch)
        self.conv0 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.sft1 = SFTLayer(in_ch, out_ch)
        self.conv1 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        fea = self.sft0(x)      # (B, out_ch, h, w)
        fea = F.relu(self.conv0(fea), inplace=True)
        fea = self.sft1((fea, x[1]))
        fea = self.conv1(fea)
        return (x[0] + fea, x[1])  # return a tuple containing features and conditions
    

class ResidualBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions.
    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.act = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        out = self.act(out)

        if self.skip is not None:
            identity = self.skip(x)

        out = out + identity
        return out
    
#+++++++++++++++++++++++++++++++++++++++ channel wise auto-regressive +++++++++++++++++++++++++++++++++++++#
# from compressai.ans import RansEncoder, RansDecoder, BufferedRansEncoder
# class CWRegressive(nn.Module):
#     def __init__(self, N, M, slice=5, training=True):
#         super().__init__()
#         assert M % slice == 0
#         # N: latent dim, M: hyper latent dim, slice, channel chunk num
#         self.training = training
#         self.slice = slice
#         self.slice_size = M // slice
#         EP_inputs = [i*self.slice_size for i in range(self.slice)]
#         self.EPlist = nn.ModuleList([])
#         for y_size in EP_inputs:
#             EP = nn.Sequential(
#                 nn.Conv2d(y_size + M, M - (N//2), stride=1, kernel_size=3, padding=3//2),
#                 nn.LeakyReLU(inplace=True),
#                 nn.Conv2d(M - (N//2), (M+N) // 4, stride=1, kernel_size=3, padding=3//2),
#                 nn.LeakyReLU(inplace=True),
#                 nn.Conv2d((M+N)//4, M*2//self.slice, stride=1, kernel_size=3, padding=3//2),
#             )
#             self.EPlist.append(EP)
        
    
#     def forward(self, y, hyper_params, gaussian_conditional):
#         y_slice = [
#             y[:, self.slice_size*i:self.slice_size*(i+1), :, :] for i in range(self.slice-1)     # B, C, H, W
#         ]
#         y_slice.append(y[:, self.slice_size*(self.slice-1):, :, :])
#         scales_hat_list, means_hat_list = [], []
#         y_hat_cumul = torch.Tensor().to(y.device)
#         for i in range(self.slice):
#             if i == 0:
#                 gaussian_param = self.EPlist[i](hyper_params)
#             else:
#                 gaussian_param = self.EPlist[i](
#                     torch.cat([hyper_params, y_hat_cumul], dim=1)
#                 )  
#             scales_hat, means_hat = gaussian_param.chunk(2, 1)
#             scales_hat_list.append(scales_hat)
#             means_hat_list.append(means_hat)
#             y_hat_sliced = gaussian_conditional.quantize(
#                 y_slice[i], "noise" if self.training else "dequantize"
#             )
#             y_hat_cumul = torch.cat([y_hat_cumul, y_hat_sliced], dim=1)
#         scales_all = torch.cat(scales_hat_list, dim=1)
#         means_all = torch.cat(means_hat_list, dim=1)
#         _, y_likelihoods = gaussian_conditional(y, scales_all, means=means_all)
#         return {
#             "y_hat": y_hat_cumul,
#             "y_likelihoods": y_likelihoods
#         }
        
#     def compress(self, y, hyper_params, gaussian_conditional):
#         encoder = BufferedRansEncoder()
#         cdf = gaussian_conditional.quantized_cdf.tolist()
#         cdf_lengths = gaussian_conditional.cdf_length.tolist()
#         offsets = gaussian_conditional.offset.tolist()
        
#         indexes_list = []
#         symbols_list = []
#         y_strings = []

#         list_sliced_y = [
#             y[:, i*self.slice_size:(i+1)*self.slice_size, :, :] for i in range(self.slice - 1)
#         ]
#         list_sliced_y.append(
#             y[:, self.slice_size*(self.slice-1):, :, :]
#         )
#         y_hat = torch.Tensor().to(y.device)
#         for i in range(self.slice):
#             y_sliced = list_sliced_y[i]
#             if i == 0:
#                 gaussian_params = self.EPlist[i](hyper_params)
#             else:
#                 gaussian_params = self.EPlist[i](
#                     torch.cat([hyper_params, y_hat], dim=1)     # along dim == C
#                 )
#             scales_hat, means_hat = gaussian_params.chunk(2, 1)
#             indexes = gaussian_conditional.build_indexes(scales_hat)
            
#             y_hat_sliced = gaussian_conditional.quantize(y_sliced, "symbols", means_hat)
#             symbols_list.extend(y_hat_sliced.reshape(-1).tolist())
#             indexes_list.extend(indexes.reshape(-1).tolist())
#             y_hat_sliced = y_hat_sliced + means_hat
#             y_hat = torch.cat([y_hat, y_hat_sliced], dim=1)
#         encoder.encode_with_indexes(
#             symbols_list, indexes_list, cdf, cdf_lengths, offsets
#         )
#         y_string = encoder.flush()
#         y_strings.append(y_string)
#         return {
#             "y_strings": y_strings
#         }
        
#     def decompress(self, y_strings, hyper_params, gaussian_conditional):
#         cdf = gaussian_conditional.quantized_cdf.tolist()
#         cdf_lengths = gaussian_conditional.cdf_length.tolist()
#         offsets = gaussian_conditional.offset.tolist()
        
#         decoder = RansDecoder()
#         decoder.set_stream(y_strings[0])
#         y_hat = torch.Tensor().to(hyper_params.device)
#         for i in range(self.slice):
#             if i == 0:
#                 gaussian_params = self.EPlist[i](hyper_params)
#             else:
#                 gaussian_params = self.EPlist[i](
#                     torch.cat([hyper_params, y_hat], dim=1)
#                 )
#             scales_sliced, means_sliced = gaussian_params.chunk(2, 1)
#             indexes_sliced = gaussian_conditional.build_indexes(scales_sliced)
#             y_sliced_hat = decoder.decode_stream(
#                 indexes_sliced.reshape(-1).tolist(), cdf, cdf_lengths, offsets
#             )
#             y_sliced_hat = torch.Tensor(y_sliced_hat).reshape(scales_sliced.shape).to(scales_sliced.device)
#             y_sliced_hat = y_sliced_hat + means_sliced
#             y_hat = torch.cat([y_hat, y_sliced_hat], dim=1)
#         return {
#             "y_hat": y_hat
#         }