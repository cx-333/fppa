
import torch
import torch.nn.functional as F
from torch import nn
from .common import CommonCompression
from .layers import SubpelConv2x, DepthConvBlock, ResidualBlockUpsample, ResidualBlockWithStride2


qp_shift = [0, 8, 4]
extra_qp = max(qp_shift)

g_ch_src_d = 3 * 8 * 8
g_ch_recon = 320
g_ch_y = 128
g_ch_z = 128
g_ch_d = 256


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
        )
        self.conv2 = nn.Sequential(
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
        )

    def forward(self, x, quant):
        x1, ctx_t = self.forward_part1(x, quant)
        ctx = self.forward_part2(x1)
        return ctx, ctx_t

    def forward_part1(self, x, quant):
        x1 = self.conv1(x)
        ctx_t = x1 * quant
        return x1, ctx_t

    def forward_part2(self, x1):
        ctx = self.conv2(x1)
        return ctx


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(g_ch_src_d, g_ch_d, 1)
        self.conv2 = nn.Sequential(
            DepthConvBlock(g_ch_d * 2, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
        )
        self.conv3 = DepthConvBlock(g_ch_d, g_ch_d)
        self.down = nn.Conv2d(g_ch_d, g_ch_y, 3, stride=2, padding=1)

        self.fuse_conv1_flag = False

    def forward(self, x, ctx, quant_step):
        feature = F.pixel_unshuffle(x, 8)
        return self.forward_torch(feature, ctx, quant_step)

    def forward_torch(self, feature, ctx, quant_step):
        feature = self.conv1(feature)
        feature = self.conv2(torch.cat((feature, ctx), dim=1))
        feature = self.conv3(feature)
        feature = feature * quant_step
        feature = self.down(feature)
        return feature
    
    
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = SubpelConv2x(g_ch_y, g_ch_d, 3, padding=1)
        self.conv1 = nn.Sequential(
            DepthConvBlock(g_ch_d * 2, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
        )
        self.conv2 = nn.Conv2d(g_ch_d, g_ch_d, 1)

    def forward(self, x, ctx, quant_step,):
        return self.forward_torch(x, ctx, quant_step)

    def forward_torch(self, x, ctx, quant_step):
        feature = self.up(x)
        feature = self.conv1(torch.cat((feature, ctx), dim=1))
        feature = self.conv2(feature)
        feature = feature * quant_step
        return feature
    
    
class ReconGeneration(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_d,     g_ch_recon),
            DepthConvBlock(g_ch_recon, g_ch_recon),
            DepthConvBlock(g_ch_recon, g_ch_recon),
            DepthConvBlock(g_ch_recon, g_ch_recon),
        )
        self.head = nn.Conv2d(g_ch_recon, g_ch_src_d, 1)

    def forward(self, x, quant_step):
        return self.forward_torch(x, quant_step)

    def forward_torch(self, x, quant_step):
        out = self.conv(x)
        out = out * quant_step
        out = self.head(out)
        out = F.pixel_shuffle(out, 8)
        # out = torch.clamp(out, 0., 1.)
        return out
    
 
class HyperEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_y, g_ch_z),
            ResidualBlockWithStride2(g_ch_z, g_ch_z),
            ResidualBlockWithStride2(g_ch_z, g_ch_z),
        )

    def forward(self, x):
        return self.conv(x)


class HyperDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            ResidualBlockUpsample(g_ch_z, g_ch_z),
            ResidualBlockUpsample(g_ch_z, g_ch_z),
            DepthConvBlock(g_ch_z, g_ch_y),
        )

    def forward(self, x):
        return self.conv(x)


class PriorFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_y * 3, g_ch_y * 3),
            DepthConvBlock(g_ch_y * 3, g_ch_y * 3),
            DepthConvBlock(g_ch_y * 3, g_ch_y * 3),
            nn.Conv2d(g_ch_y * 3, g_ch_y * 3, 1),
        )

    def forward(self, x):
        return self.conv(x)


class SpatialPrior(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_y * 4, g_ch_y * 3),
            DepthConvBlock(g_ch_y * 3, g_ch_y * 3),
            nn.Conv2d(g_ch_y * 3, g_ch_y * 2, 1),
        )

    def forward(self, x):
        return self.conv(x)


class RefFrame():
    def __init__(self):
        self.frame = None
        self.feature = None
        self.poc = None
        

class DCVCRTVideo(CommonCompression):
    def __init__(self, z_channel=128, extra_qp=extra_qp):
        super().__init__(z_channel, extra_qp)
        self.qp_shift = qp_shift

        self.feature_adaptor_i = DepthConvBlock(g_ch_src_d, g_ch_d)
        self.feature_adaptor_p = nn.Conv2d(g_ch_d, g_ch_d, 1)
        self.feature_extractor = FeatureExtractor()

        self.encoder = Encoder()
        self.hyper_encoder = HyperEncoder()
        self.hyper_decoder = HyperDecoder()
        self.temporal_prior_encoder = ResidualBlockWithStride2(g_ch_d, g_ch_y * 2)
        self.y_prior_fusion = PriorFusion()
        self.y_spatial_prior = SpatialPrior()
        self.decoder = Decoder()
        self.recon_generation_net = ReconGeneration()

        self.q_encoder = nn.Parameter(torch.ones((self.get_qp_num() + extra_qp, g_ch_d, 1, 1)))
        self.q_decoder = nn.Parameter(torch.ones((self.get_qp_num() + extra_qp, g_ch_d, 1, 1)))
        self.q_feature = nn.Parameter(torch.ones((self.get_qp_num() + extra_qp, g_ch_d, 1, 1)))
        self.q_recon = nn.Parameter(torch.ones((self.get_qp_num() + extra_qp, g_ch_recon, 1, 1)))

        self.dpb = []
        self.max_dpb_size = 1
        self.curr_poc = 0
        
    def reset_ref_feature(self):
        if len(self.dpb) > 0:
            self.dpb[0].feature = None

    def add_ref_frame(self, feature=None, frame=None, increase_poc=True):
        ref_frame = RefFrame()
        ref_frame.poc = self.curr_poc           # type: ignore
        ref_frame.frame = frame
        ref_frame.feature = feature
        if len(self.dpb) >= self.max_dpb_size:
            self.dpb.pop(-1)
        self.dpb.insert(0, ref_frame)
        if increase_poc:
            self.curr_poc += 1

    def clear_dpb(self):
        self.dpb.clear()

    def set_curr_poc(self, poc):
        self.curr_poc = poc

    def apply_feature_adaptor(self):
        if self.dpb[0].feature is None:
            return self.feature_adaptor_i(F.pixel_unshuffle(self.dpb[0].frame, 8))
        return self.feature_adaptor_p(self.dpb[0].feature)

    def res_prior_param_decoder(self, z_hat, ctx_t):
        hierarchical_params = self.hyper_decoder(z_hat)
        temporal_params = self.temporal_prior_encoder(ctx_t)
        _, _, H, W = temporal_params.shape
        hierarchical_params = hierarchical_params[:, :, :H, :W].contiguous()
        params = self.y_prior_fusion(
            torch.cat((hierarchical_params, temporal_params), dim=1))
        return params

    def get_recon_and_feature(self, y_hat, ctx, q_decoder, q_recon):
        feature = self.decoder(y_hat, ctx, q_decoder)
        x_hat = self.recon_generation_net(feature, q_recon)
        return x_hat, feature

    def prepare_feature_adaptor_i(self, last_qp):
        if self.dpb[0].frame is None:
            q_recon = self.q_recon[last_qp:last_qp+1, :, :, :]
            self.dpb[0].frame = self.recon_generation_net(self.dpb[0].feature, q_recon).clamp_(0, 1)
            self.reset_ref_feature()
            
    def forward(self, x, qp):
        # qp \in [0, ..., 63]
        q_encoder = self.q_encoder[qp]
        q_decoder = self.q_decoder[qp]
        q_feature = self.q_feature[qp]
        q_recon = self.q_recon[qp]
        
        feature = self.apply_feature_adaptor()
        ctx, ctx_t = self.feature_extractor(feature, q_feature)
        y = self.encoder(x, ctx, q_encoder)
        hyper_inp = self.pad_for_y(y)
        
        z = self.hyper_encoder(hyper_inp)
        z_hat = self.quant(z)
        # z_hat, z_likelihoods = self.entropy_bottleneck(z, qp)
        # z_bits = self.bit_estimator_z.get_z_bits(z, qp)
        # z_hat = self.quant(z)
        
        params = self.res_prior_param_decoder(z_hat, ctx_t)
        
        y_res, y_hat, y_q, y_scales_hat = self.forward_bintree_prior(y, params, self.y_spatial_prior)
        
        x_hat, feature = self.get_recon_and_feature(y_hat, ctx, q_decoder, q_recon)
        
        x_hat = torch.clamp(x_hat, 0, 1.0)
        
        self.add_ref_frame(feature, x_hat)
        
        
        if self.training:
            y_for_bit = self.add_noise(y_res)
            z_for_bit = self.add_noise(z)
        else:
            y_for_bit = y_q 
            z_for_bit = z_hat
        
        # NOTE: OpenDCVCs; log scale
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
            "bpp": bpp,
        }
        
    @torch.no_grad()
    def compress(self, x, qp):
        q_encoder = self.q_encoder[qp]
        q_feature = self.q_feature[qp]
        
        # 1. 提取当前 DPB 参考帧的上下文特征
        feature = self.apply_feature_adaptor()
        ctx, ctx_t = self.feature_extractor(feature, q_feature)
        
        # 2. 核心特征编码
        y = self.encoder(x, ctx, q_encoder)
        hyper_inp = self.pad_for_y(y)
        _, _, yH, yW = y.size()
        
        # 3. 超先验提取与量化
        z = self.hyper_encoder(hyper_inp)
        z_hat = self.quant(z)
        
        _, _, zH, zW = z_hat.size()
        
        # 4. 压缩超先验 z
        self.entropy_coder.reset()
        self.bit_estimator_z.encode_z(z_hat, qp)
        self.entropy_coder.flush()
        z_string = self.entropy_coder.get_encoded_stream()
        
        # 5. 从 z_hat 与 时序先验 ctx_t 中解码 params
        params = self.res_prior_param_decoder(z_hat, ctx_t)
        
        # 6. 压缩核心特征 y (使用二叉树自回归编码)
        y_comp_out = self.compress_bintree_prior(y, params, self.y_spatial_prior)
        y_string = y_comp_out["y_strings"][0]
        
        return {
            "strings": [y_string, z_string],
            "shape": [zH, zW]
        }

    @torch.no_grad()
    def decompress(self, y_string, z_string, shape, qp):
        device = next(self.parameters()).device
        q_decoder = self.q_decoder[qp]
        q_feature = self.q_feature[qp]
        q_recon = self.q_recon[qp]
        
        zH, zW = shape
        
        # 1. 提取当前 DPB 参考帧的上下文特征 (必须与压缩端保持完全一致)
        feature_dpb = self.apply_feature_adaptor()
        ctx, ctx_t = self.feature_extractor(feature_dpb, q_feature)
        
        self.entropy_coder.set_stream(z_string)
        self.bit_estimator_z.decode_z((zH, zW), qp)
        z_hat = self.bit_estimator_z.get_z((zH, zW), device, q_decoder.dtype)
        
        # 3. 解码二叉树所需的 params
        params = self.res_prior_param_decoder(z_hat, ctx_t)
        
        # 4. 解压特征 y
        y_hat = self.decompress_bintree_prior([y_string], params, self.y_spatial_prior)
        
        # 5. 生成最终重构帧并更新时序参考帧 DPB (极其关键)
        x_hat, feature = self.get_recon_and_feature(y_hat, ctx, q_decoder, q_recon)
        self.add_ref_frame(feature, x_hat)
        
        return {
            "x_hat": x_hat,
            "y_hat": y_hat,
            "feature": feature
        }
        
        