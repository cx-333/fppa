

from .entropy_net import BitEstimator, GaussianEncoder, EntropyCoder
import math
import torch
from torch import nn 
from typing import Any, Mapping
import torch.nn.functional as F


def replicated_pad(x, pad_b, pad_r):
    if pad_b == 0 and pad_r == 0:
        return x
    return F.pad(x, (0, pad_r, 0, pad_b), mode="replicate")


class CommonCompression(nn.Module):
    def __init__(self, z_channel, extra_qp=0):
        super().__init__()
        self.z_channel = z_channel
        self.bit_estimator_z = BitEstimator(64+extra_qp, z_channel)
        self.gaussian_encoder = GaussianEncoder()
        
        self.force_zero_thres = 0.12
        
        # 4x 模式的 4 种基础图案
        self.register_buffer("m_pat_4x_0", torch.tensor([[1, 0], [0, 0]], dtype=torch.float32).view(1, 1, 2, 2))
        self.register_buffer("m_pat_4x_1", torch.tensor([[0, 1], [0, 0]], dtype=torch.float32).view(1, 1, 2, 2))
        self.register_buffer("m_pat_4x_2", torch.tensor([[0, 0], [1, 0]], dtype=torch.float32).view(1, 1, 2, 2))
        self.register_buffer("m_pat_4x_3", torch.tensor([[0, 0], [0, 1]], dtype=torch.float32).view(1, 1, 2, 2))
        
        # 2x 模式的 2 种基础图案
        self.register_buffer("m_pat_2x_0", torch.tensor([[1, 0], [0, 1]], dtype=torch.float32).view(1, 1, 2, 2))
        self.register_buffer("m_pat_2x_1", torch.tensor([[0, 1], [1, 0]], dtype=torch.float32).view(1, 1, 2, 2))
        
        self.cuda_streams = {} 
        
        
    def get_cuda_stream(self, device, idx=0, priority=0):
        key = f"{device}_{priority}_{idx}"
        if key not in self.cuda_streams:
            self.cuda_streams[key] = torch.cuda.Stream(device, priority=priority)
        return self.cuda_streams[key]      
        
    @staticmethod
    def get_qp_num():
        return 64
        
    def _get_spatial_mask_from_pattern(self, pattern_buffer, height, width, dtype):
        """
        利用常驻显存的 pattern_buffer 动态平铺出所需的空间尺寸 (1, 1, H, W)
        """
        # 转为目标数据类型 (float16/float32)，由于在同显卡，转换极快
        mask = pattern_buffer.to(dtype)
        # 空间上重复平铺
        mask = mask.repeat(1, 1, (height + 1) // 2, (width + 1) // 2)
        # 裁剪掉多余的边缘
        return mask[:, :, :height, :width]
    
    def update(self, force=True, force_zero_thres=None):
        self.entropy_coder = EntropyCoder()
        self.gaussian_encoder.update(force, self.entropy_coder, force_zero_thres=force_zero_thres)
        self.bit_estimator_z.update(force, self.entropy_coder)
    
    def set_use_two_entropy_coders(self, use_two_entropy_coders):
        self.entropy_coder.set_use_two_entropy_coders(use_two_entropy_coders)
    
    def pad_for_y(self, y):
        _, _, H, W = y.size()
        padding_r, padding_b = self.get_padding_size(H, W, 4)
        y_pad = replicated_pad(y, padding_b, padding_r)
        return y_pad
        
    def quant(self, x):
        if self.training:
            n = torch.round(x) - x 
            n = n.clone().detach()
            return x + n
        # if self.training:
        #     noise = torch.empty_like(x).uniform_(-0.5, 0.5)
        #     return x + noise
        out = torch.round(x)
        return out 
    
    def ste_round(self, x):
        if self.training:
            n = torch.round(x) - x 
            n = n.clone().detach()
            return x + n 
        return torch.round(x)
    
    def add_noise(self, x):
        noise = torch.empty_like(x).uniform_(-0.5, 0.5)
        return x + noise

    @staticmethod
    def get_padding_size(height, width, p=64):
        new_h = (height + p - 1) // p * p
        new_w = (width + p - 1) // p * p
        padding_right = new_w - width
        padding_bottom = new_h - height
        return padding_right, padding_bottom
    
    @staticmethod
    def get_downsampled_shape(height, width, p):
        new_h = (height + p - 1) // p * p
        new_w = (width + p - 1) // p * p
        return int(new_h / p + 0.5), int(new_w / p + 0.5)
    
    @staticmethod
    def get_one_mask(micro_mask, height, width, dtype, device):
        mask = torch.tensor(micro_mask, dtype=dtype, device=device) # 2x2
        mask = mask.repeat((height+1)//2, (width+1)//2)
        mask = mask[:height, :width]    # height x width
        mask = torch.unsqueeze(mask, 0)
        mask = torch.unsqueeze(mask, 0) # 1 x 1 x height x width
        return mask

    def get_mask_4x(self, batch, channel, height, width, dtype, device):
        # 取消了重新分配 torch.ones，改为全视图操作 (View & Expand)
        with torch.no_grad():
            assert channel % 4 == 0
            sub_c = channel // 4
            
            # 1. 拿到 1x1xHxW 的空间掩码
            m0 = self._get_spatial_mask_from_pattern(self.m_pat_4x_0, height, width, dtype)
            m1 = self._get_spatial_mask_from_pattern(self.m_pat_4x_1, height, width, dtype)
            m2 = self._get_spatial_mask_from_pattern(self.m_pat_4x_2, height, width, dtype)
            m3 = self._get_spatial_mask_from_pattern(self.m_pat_4x_3, height, width, dtype)

            # 2. 利用 expand 瞬间“投影”到 batch 和 sub_c 维度 (零显存开销！)
            m0_exp = m0.expand(batch, sub_c, height, width)
            m1_exp = m1.expand(batch, sub_c, height, width)
            m2_exp = m2.expand(batch, sub_c, height, width)
            m3_exp = m3.expand(batch, sub_c, height, width)

            # 3. 在 Channel 维度上交错拼接
            mask_0 = torch.cat((m0_exp, m1_exp, m2_exp, m3_exp), dim=1)
            mask_1 = torch.cat((m3_exp, m2_exp, m1_exp, m0_exp), dim=1)
            mask_2 = torch.cat((m2_exp, m3_exp, m0_exp, m1_exp), dim=1)
            mask_3 = torch.cat((m1_exp, m0_exp, m3_exp, m2_exp), dim=1)
            
        return [mask_0, mask_1, mask_2, mask_3]

    def get_mask_2x(self, batch, channel, height, width, dtype, device):
        with torch.no_grad():
            assert channel % 2 == 0
            sub_c = channel // 2
            
            m0 = self._get_spatial_mask_from_pattern(self.m_pat_2x_0, height, width, dtype)
            m1 = self._get_spatial_mask_from_pattern(self.m_pat_2x_1, height, width, dtype)
            
            m0_exp = m0.expand(batch, sub_c, height, width)
            m1_exp = m1.expand(batch, sub_c, height, width)
            
            mask_0 = torch.cat((m0_exp, m1_exp), dim=1)
            mask_1 = torch.cat((m1_exp, m0_exp), dim=1)
            
        return [mask_0, mask_1]


    @staticmethod
    def single_part_for_writing_4x(x):
        """
        1010    0000    0101    0000    1111
        0000    0101    0000    1010    1111
        """
        x0, x1, x2, x3 = x.chunk(4, 1)
        return (x0 + x1) + (x2 + x3)
    
    @staticmethod
    def single_part_for_writing_2x(x):
        x0, x1 = x.chunk(2, 1)
        return x0 + x1
    

    def separate_prior(self, params, is_video=False):
        if is_video:
            quant_step, scales, means = params.chunk(3, 1)
            quant_step = torch.clamp_min(quant_step, 0.5)   #
            # quant_step = torch.sigmoid(quant_step) * 1.5 + 0.5  # >= 0.5
            q_enc = 1. / quant_step
            q_dec = quant_step
        else:
            q = params[:, :2, :, :]
            q_enc, q_dec = (torch.sigmoid(q) * 1.5 + 0.5).chunk(2, 1)
            scales, means = params[:, 2:, :, :].chunk(2, 1)
        return q_enc, q_dec, scales, means
    
    
    def process_with_mask(self, y, scales, means, mask):
        # 乘以掩码，提取当前需要处理的上下文特征
        scales_hat = scales * mask
        means_hat = means * mask
        y_res = (y - means_hat) * mask
        
        # 根据模型是否处于训练模式，选择不同的量化方法
        # if self.training:
        #     # NOTE: OpenDCVCs 使用STE量化 + noise
        #     # 训练模式：使用均匀噪声模拟量化误差，梯度可以平滑回传
        #     noise = torch.empty_like(y_res).uniform_(-0.5, 0.5)
        #     y_q = y_res + noise
        # else:
        #     # 推理/测试模式：使用硬舍入，得到真实的量化值
        #     y_q = torch.round(y_res)
        # 仅在推理时应用零值化阈值，以跳过对高概率为零的系数的编码
        # if not self.training and self.force_zero_thres is not None:
        #     cond = scales_hat > self.force_zero_thres
        #     # 将条件转为浮点数掩码相乘
        #     y_q = y_q * cond.float() 
        # 极值保护，防止算术编码器查表越界
        # if not self.training:
        #     y_q = torch.clamp(y_q, -128., 127.)
        
        y_q = self.quant(y_res)
        y_q = torch.clamp(y_q, -128., 127.)
        # 加上均值，完成当前掩码部分的特征重构
        y_hat = y_q + means_hat
        
        return y_res, y_q, y_hat, scales_hat


    @staticmethod
    def restore_y_4x(y, means, mask):
        return (torch.cat((y, y, y, y), dim=1) + means) * mask
    
    @staticmethod
    def restore_mask_4x(y, mask):
        return torch.cat((y, y, y, y), dim=1) * mask
    
    @staticmethod
    def restore_mask_2x(y, mask):
        return torch.cat((y, y), dim=1) * mask
    
    def forward_quadtree_prior(self, y, params, y_spatial_prior_reduction, y_spatial_prior_adaptors, y_spatial_prior):
        # y: (B, N, h, w)
        # params: (B, N*2+2, h, w)
        
        y_hat = torch.zeros_like(y)
        # y_likelihoods = torch.zeros_like(y)
        
        dtype, device = y.dtype, y.device
        B, C, H, W = y.size()
        masks = self.get_mask_4x(B, C, H, W, dtype, device)
        
        # step 1. q_enc, q_dec, scales, means
        q_enc, q_dec, scales, means = self.separate_prior(params)
        common_params = y_spatial_prior_reduction(params)   # (B, N, h, w)
        
        scales_hat = []
        y_qs = []
        y_res_hat = []
        
        y = y * q_enc
        for i, mask in enumerate(masks):
            
            y_res_0, y_q_0, y_hat_0, s_hat_0 = self.process_with_mask(y, scales, means, mask)
            scales_hat.append(s_hat_0)
            y_qs.append(y_q_0)
            y_res_hat.append(y_res_0)
            
            y_hat = y_hat + y_hat_0
            ctx = torch.cat((y_hat, common_params), dim=1)
            if i < len(masks) - 1:
                scales, means = y_spatial_prior(y_spatial_prior_adaptors[i](ctx)).chunk(2, 1)       # update prior info
            
            # y_q_w = self.single_part_for_writing_4x(y_q_0)
            # s_w = self.single_part_for_writing_4x(s_hat_0)
            # # 不能调用 _, y_indices_likelihoods = self.gaussian_conditional(y_q_w, s_w); 会进行二次加噪，影响模型训练
            # y_indices_likelihoods = self.gaussian_encoder.get_y_bits(y_q_w, s_w)       
            # y_likelihoods = y_likelihoods + self.restore_mask_4x(y_indices_likelihoods, mask)
        
        y_hat = y_hat * q_dec

        y_scales_hat = torch.zeros_like(y)
        y_q = torch.zeros_like(y)
        y_res = torch.zeros_like(y)
        for i in range(len(masks)):
            y_scales_hat = y_scales_hat + scales_hat[i]
            y_q = y_q + y_qs[i]
            y_res = y_res + y_res_hat[i]
        # y_hat, 
        return y_res, y_hat, y_q, y_scales_hat
    
    
    def forward_bintree_prior(self, y, params, y_spatial_prior):
        # params: (B, g_ch_y*3, h, w)
        # y, q_dec, scales, means: (B, g_ch_y, h, w)
        q_enc, q_dec, scales, means = self.separate_prior(params, is_video=True)
        
        y_hat = torch.zeros_like(y)
        y_likelihoods = torch.zeros_like(y)
        
        dtype, device = y.dtype, y.device
        B, C, H, W = y.size()
        masks = self.get_mask_2x(B, C, H, W, dtype, device)
        
        scales_hat, y_qs = [], []
        y_res_hat = []
        
        y = y * q_enc
        for i, mask in enumerate(masks):
            
            y_res_0, y_q_0, y_hat_0, s_hat_0 = self.process_with_mask(y, scales, means, mask)
            scales_hat.append(s_hat_0)
            y_qs.append(y_q_0)
            y_res_hat.append(y_res_0)
           
            y_hat = y_hat + y_hat_0
            ctx = torch.cat((y_hat_0, params), dim=1)
            if i < len(masks) - 1:
                scales, means = y_spatial_prior(ctx).chunk(2, 1)    # update prior info   
                    
            # y_q_w = self.single_part_for_writing_2x(y_q_0)
            # s_w = self.single_part_for_writing_2x(s_hat_0)
            # y_indices_likelihoods = self.gaussian_encoder.get_y_bits(y_q_w, s_w)       
            # y_likelihoods = y_likelihoods + self.restore_mask_2x(y_indices_likelihoods, mask)
        
        y_hat = y_hat * q_dec
        
        y_scales_hat = torch.zeros_like(y)
        y_q = torch.zeros_like(y)
        y_res = torch.zeros_like(y)
        for i in range(len(masks)):
            y_scales_hat = y_scales_hat + scales_hat[i]
            y_q = y_q + y_qs[i]
            y_res = y_res + y_res_hat[i]
        return y_res, y_hat, y_q, y_scales_hat
        
    
    def compress_quadtree_prior(self, y, params, y_spatial_prior_reduction, y_spatial_prior_adaptors, y_spatial_prior):
        # y: (B, N, h, w)
        # params: (B, N*2+2, h, w)
        B, C, H, W = y.size()
        dtype, device = y.dtype, y.device
        y_hat = torch.zeros_like(y)

        masks = self.get_mask_4x(B, C, H, W, dtype, device)
        q_enc, q_dec, scales, means = self.separate_prior(params)
        common_params = y_spatial_prior_reduction(params)   # (B, N, h, w)
        y = y * q_enc

        self.entropy_coder.reset()

        for i, mask in enumerate(masks):
            # 在压缩时，模型应处于 eval 模式，process_with_mask 内部会使用 torch.round
            y_res_0, y_q_0, y_hat_0, s_hat_0 = self.process_with_mask(y, scales, means, mask)
            y_hat = y_hat + y_hat_0
            ctx = torch.cat((y_hat, common_params), dim=1)
            if i < len(masks) - 1:
                scales, means = y_spatial_prior(y_spatial_prior_adaptors[i](ctx)).chunk(2, 1)       # update prior info
            
            y_q_w = self.single_part_for_writing_4x(y_q_0)
            s_w = self.single_part_for_writing_4x(s_hat_0)

            self.gaussian_encoder.encode_y(y_q_w, s_w)

        self.entropy_coder.flush()
        y_string = self.entropy_coder.get_encoded_stream()
        
        return {"y_strings": [y_string]}
    

    def decompress_quadtree_prior(self, y_strings, params, y_spatial_prior_reduction, y_spatial_prior_adaptors, y_spatial_prior):
        if not isinstance(y_strings, list) or len(y_strings) != 1:
            raise ValueError("Expected a list containing a single bitstream string for the entire batch.")

        y_string = y_strings[0]
        self.entropy_coder.set_stream(y_string)

        _, q_dec, scales, means = self.separate_prior(params)
        B, C, H, W = means.size()
        dtype, device = means.dtype, means.device
        
        y_hat = torch.zeros((B, C, H, W), dtype=dtype, device=device)
        masks = self.get_mask_4x(B, C, H, W, dtype, device)
        common_params = y_spatial_prior_reduction(params)   # (B, N, h, w)
        
        for i, mask in enumerate(masks):
            s_hat_0 = scales * mask
            m_hat_0 = means * mask

            s_w = self.single_part_for_writing_4x(s_hat_0)
            
            y_q_w = self.gaussian_encoder.decode_and_get_y(s_w, dtype, device)
            
            y_q_0 = self.restore_mask_4x(y_q_w, mask)
            
            y_hat_0_step = y_q_0 + m_hat_0
            y_hat = y_hat + y_hat_0_step
            
            if i < len(masks) - 1:
                ctx = torch.cat((y_hat, common_params), dim=1)
                scales, means = y_spatial_prior(y_spatial_prior_adaptors[i](ctx)).chunk(2, 1)
                
        y_hat = y_hat * q_dec
        return y_hat
    
    def compress_bintree_prior(self, y, params, y_spatial_prior):
        B, C, H, W = y.size()
        dtype, device = y.dtype, y.device
        y_hat = torch.zeros_like(y)
        
        masks = self.get_mask_2x(B, C, H, W, dtype, device)
        q_enc, q_dec, scales, means = self.separate_prior(params, is_video=True)
        y = y * q_enc
        
        self.entropy_coder.reset()

        for i, mask in enumerate(masks):
            # 在压缩时，模型应处于 eval 模式，process_with_mask 内部会使用 torch.round
            y_res_0, y_q_0, y_hat_0, s_hat_0 = self.process_with_mask(y, scales, means, mask)
            y_hat = y_hat + y_hat_0
            ctx = torch.cat((y_hat_0, params), dim=1)
            if i < len(masks) - 1:
                scales, means = y_spatial_prior(ctx).chunk(2, 1)    # update prior info
            
            y_q_w = self.single_part_for_writing_2x(y_q_0)
            s_w = self.single_part_for_writing_2x(s_hat_0)

            self.gaussian_encoder.encode_y(y_q_w, s_w)

        self.entropy_coder.flush()
        y_string = self.entropy_coder.get_encoded_stream()
        
        return {"y_strings": [y_string]}

    def decompress_bintree_prior(self, y_strings, params, y_spatial_prior):
        if not isinstance(y_strings, list) or len(y_strings) != 1:
            raise ValueError("Expected a list containing a single bitstream string for the entire batch.")

        y_string = y_strings[0]
        self.entropy_coder.set_stream(y_string)

        _, q_dec, scales, means = self.separate_prior(params, is_video=True)
        B, C, H, W = means.size()
        dtype, device = means.dtype, means.device
        
        y_hat = torch.zeros((B, C, H, W), dtype=dtype, device=device)
        masks = self.get_mask_2x(B, C, H, W, dtype, device)
        
        for i, mask in enumerate(masks):
            s_hat_0 = scales * mask
            m_hat_0 = means * mask

            s_w = self.single_part_for_writing_2x(s_hat_0)
            
            y_q_w = self.gaussian_encoder.decode_and_get_y(s_w, dtype, device)
            
            y_q_0 = self.restore_mask_2x(y_q_w, mask)
            y_hat_0_step = y_q_0 + m_hat_0
            
            y_hat = y_hat + y_hat_0_step
            
            if i < len(masks) - 1:
                ctx = torch.cat((y_hat_0_step, params), dim=1)
                scales, means = y_spatial_prior(ctx).chunk(2, 1)
                
        y_hat = y_hat * q_dec
        return y_hat
    
    