


from .layers import DepthConvBlock, ResidualBlockUpsample, ResidualBlockWithStride2
import torch
from torch import nn
import torch.nn.functional as F
from .common import CommonCompression
import math
import numpy as np
from functools import partial



g_ch_src = 3*8*8
g_ch_enc_dec = 368


class IntraEncoder(nn.Module):
    def __init__(self, N):
        super().__init__()

        self.enc_1 = DepthConvBlock(g_ch_src, g_ch_enc_dec)
        self.enc_2 = nn.Sequential(
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),

            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec), # 2.2
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec), # 2.5
            nn.Conv2d(g_ch_enc_dec, N, 3, stride=2, padding=1),
        )

    def forward(self, x, quant_step):
        feature = F.pixel_unshuffle(x, 8)
        return self.forward_torch(feature, quant_step), feature

    def forward_torch(self, out, quant_step):
        out = self.enc_1(out)
        out = out * quant_step
        return self.enc_2(out)
    

class IntraDecoder(nn.Module):
    def __init__(self, N):
        super().__init__()

        self.dec_1 = nn.Sequential(
            ResidualBlockUpsample(N, g_ch_enc_dec),         # 0
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec), 
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),     # 9
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),     # 10
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
        )
        self.dec_2 = DepthConvBlock(g_ch_enc_dec, g_ch_src)

    def forward(self, x, quant_step):
        return self.forward_torch(x, quant_step)

    def forward_torch(self, x, quant_step):
        out = self.dec_1(x)
        out = out * quant_step
        feature = self.dec_2(out)
        out = F.pixel_shuffle(feature, 8)
        return out, feature
    

class DCVCRTImage(CommonCompression):
    def __init__(self, N=256, z_channel=128):
        super().__init__(z_channel)
        
        self.enc = IntraEncoder(N)
        
        self.hyper_enc = nn.Sequential(
            DepthConvBlock(N, z_channel),
            ResidualBlockWithStride2(z_channel, z_channel),
            ResidualBlockWithStride2(z_channel, z_channel),
        )

        self.hyper_dec = nn.Sequential(
            ResidualBlockUpsample(z_channel, z_channel),
            ResidualBlockUpsample(z_channel, z_channel),
            DepthConvBlock(z_channel, N),
        )
        
        self.y_prior_fusion = nn.Sequential(
            DepthConvBlock(N, N * 2),
            DepthConvBlock(N * 2, N * 2),
            DepthConvBlock(N * 2, N * 2),
            nn.Conv2d(N * 2, N * 2 + 2, 1),
        )
        
        self.y_spatial_prior_reduction = nn.Conv2d(N * 2 + 2, N * 1, 1)
        
        self.y_spatial_prior_adaptors = nn.ModuleList([
            DepthConvBlock(N * 2, N * 2, force_adaptor=True) for _ in range(3)
        ])
        
        self.y_spatial_prior = nn.Sequential(
            DepthConvBlock(N * 2, N * 2),
            DepthConvBlock(N * 2, N * 2),
            DepthConvBlock(N * 2, N * 2),
            nn.Conv2d(N * 2, N * 2, 1),
        )
        
        self.dec = IntraDecoder(N)
        
        self.q_scale_enc = nn.Parameter(torch.ones((self.get_qp_num(), g_ch_enc_dec, 1, 1)))
        self.q_scale_dec = nn.Parameter(torch.ones((self.get_qp_num(), g_ch_enc_dec, 1, 1)))
        


    def forward(self, x, qp): 
        # qp \in [0, ..., 63]
        
        # device = x.device
        curr_q_enc = self.q_scale_enc[qp]
        curr_q_dec = self.q_scale_dec[qp]
        
        # analysis
        y, feature = self.enc(x, curr_q_enc)
        y_pad = self.pad_for_y(y)
        
        # hyper encoder and decoder
        z = self.hyper_enc(y_pad)
        # ste 
        z_hat = self.quant(z)
        
        params = self.hyper_dec(z_hat)      # (B, N, h, w)
        _, _, yH, yW = y.shape
        params = params[:, :, :yH, :yW].contiguous()
        
        params = self.y_prior_fusion(params)
        
        # Spatial prior processing using 4x method
        # self.y_spatial_prior_reduction, self.y_spatial_prior_adaptors, self.y_spatial_prior 
        y_res, y_hat, y_q, y_scales_hat = self.forward_quadtree_prior(y, \
                                            params, self.y_spatial_prior_reduction, \
                                            self.y_spatial_prior_adaptors, self.y_spatial_prior)
        
        x_hat, feature_hat = self.dec(y_hat, curr_q_dec)
        
        x_hat = torch.clamp(x_hat, 0.0, 1.0)
        
        
        if self.training:
            y_for_bit = self.add_noise(y_res)
            z_for_bit = self.add_noise(z)
        else:
            y_for_bit = y_q
            z_for_bit = z_hat
        
        # NOTE: OpenDCVCs, log scale
        # y_scales_hat = torch.nn.functional.softplus(y_scales_hat+2.3)-2.3 #make logscale > -2.3
        # y_scales_hat = 5.55 - torch.nn.functional.softplus(5.55 - y_scales_hat)  # make logscale < 5.55
        # y_scales_hat = torch.exp(y_scales_hat)
        
        _, _, H, W = x.size()
        pixel_num = H * W 
        bits_y = self.gaussian_encoder.get_y_bits(y_for_bit, y_scales_hat)
        bits_z = self.bit_estimator_z.get_z_bits(z_for_bit, qp)
        bpp_y = torch.sum(bits_y, dim=[1, 2, 3]) / pixel_num 
        bpp_z = torch.sum(bits_z, dim=[1, 2, 3]) / pixel_num
        bpp = bpp_y + bpp_z
        
        return {
            "x_hat": x_hat,
            "y_hat": y_hat,
            "bpp": bpp, 
        }  
        
    @torch.no_grad()
    def compress(self, x, qp):
        curr_q_enc = self.q_scale_enc[qp]
        
        y, _ = self.enc(x, curr_q_enc)
        _, _, yH, yW = y.size()
        
        y_pad = self.pad_for_y(y)
        z = self.hyper_enc(y_pad) 
        z_hat = self.quant(z)
        _, _, zH, zW = z_hat.size()
        
        # 1. 压缩超先验 z
        self.entropy_coder.reset()
        self.bit_estimator_z.encode_z(z_hat, qp)
        self.entropy_coder.flush()
        z_string = self.entropy_coder.get_encoded_stream()
        
        # 2. 从 z_hat 解码生成四叉树所需的 params
        params = self.hyper_dec(z_hat)
        params = params[:, :, :yH, :yW].contiguous()
        params = self.y_prior_fusion(params)
        
        # 3. 压缩核心特征 y (复用 CommonCompression 内置四叉树串行编码)
        y_comp_out = self.compress_quadtree_prior(
            y, params,
            self.y_spatial_prior_reduction,
            self.y_spatial_prior_adaptors,
            self.y_spatial_prior
        )
        y_string = y_comp_out["y_strings"][0]
        
        return {
            "strings": [y_string, z_string],
            "shape": [zH, zW]
        }

    @torch.no_grad()
    def decompress(self, y_string, z_string, shape, qp, enhance_mode='balanced', rec_weight=0.5):
        device = next(self.parameters()).device
        curr_q_dec = self.q_scale_dec[qp]
        
        zH, zW = shape
        
        # 2. 解压超先验 z
        self.entropy_coder.set_stream(z_string)
        self.bit_estimator_z.decode_z((zH, zW), qp)
        z_hat = self.bit_estimator_z.get_z((zH, zW), device, curr_q_dec.dtype)
        
        # 3. 生成四叉树先验 params
        params = self.hyper_dec(z_hat)
        params = params[:, :, :zH*4, :zW*4].contiguous()
        params = self.y_prior_fusion(params)
        
        # 4. 解压特征 y
        y_hat = self.decompress_quadtree_prior(
            [y_string], params,
            self.y_spatial_prior_reduction,
            self.y_spatial_prior_adaptors,
            self.y_spatial_prior
        )
        
        x_hat, feature_hat = self.dec(y_hat, curr_q_dec)
        
        x_hat = torch.clamp(x_hat, 0.0, 1.0)
            
        return {
            "x_hat": x_hat,
            "y_hat": y_hat,
            "feature": feature_hat
        }
        
        