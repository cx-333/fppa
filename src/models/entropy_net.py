import torch 
import math
import numpy as np
from torch import nn 
import torch.nn.functional as F 


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


class EntropyCoder(object):
    def __init__(self):
        super().__init__()
        
        from MLCodec_extensions_cpp import RansEncoder, RansDecoder 
        self.encoder = RansEncoder()
        self.decoder = RansDecoder() 
        
    @staticmethod
    def pmf_to_quantized_cdf(pmf, precision=16):
        # cumulative distribution function; equation: CDF(x) = sum_{i=0}^{x} p(i)
        # quantized CDF: CDF(x) = round(CDF(x) * 2^precision)
        from MLCodec_extensions_cpp import pmf_to_quantized_cdf as _pmf_to_cdf
        cdf = _pmf_to_cdf(pmf.tolist(), precision)
        cdf = torch.IntTensor(cdf)
        return cdf          

    @staticmethod
    def pmf_to_cdf(pmf, tail_mass, pmf_length, max_length):
        entropy_coder_precision = 16
        cdf = torch.zeros((len(pmf_length), max_length + 2), dtype=torch.int32)
        for i, p in enumerate(pmf):
            prob = torch.cat((p[: pmf_length[i]], tail_mass[i]), dim=0)
            _cdf = EntropyCoder.pmf_to_quantized_cdf(prob, entropy_coder_precision)
            cdf[i, : _cdf.size(0)] = _cdf
        return cdf 
    
    def reset(self):
        self.encoder.reset()
        
    def add_cdf(self, cdf, cdf_length, offset):
        enc_cdf_idx = self.encoder.add_cdf(cdf, cdf_length, offset)
        dec_cdf_idx = self.decoder.add_cdf(cdf, cdf_length, offset)
        assert enc_cdf_idx == dec_cdf_idx
        return enc_cdf_idx
    
    def encode_y(self, symbols, cdf_group_index):
        # symbols: int16, high 8 bits: int8 symbol to be encoded; low 8 bits: uint8 index to use
        assert symbols.dtype == torch.int16
        self.encoder.encode_y(symbols.cpu().numpy(), cdf_group_index)
        
    def encode_z(self, symbols, cdf_group_index, start_offset, per_channel_size):
        self.encoder.encode_z(symbols.to(torch.int8).cpu().numpy(),
                              cdf_group_index, start_offset, per_channel_size)
        
    def flush(self):
        self.encoder.flush()
        
    def get_encoded_stream(self):
        return self.encoder.get_encoded_stream().tobytes()
    
    def set_stream(self, stream):
        self.decoder.set_stream((np.frombuffer(stream, dtype=np.uint8)))
        
    def decode_y(self, indexes, cdf_group_index):
        self.decoder.decode_y(indexes.to(torch.uint8).cpu().numpy(), cdf_group_index)
        
    def decode_and_get_y(self, indexes, cdf_group_index, device, dtype):
        rv = self.decoder.decode_and_get_y(indexes.to(torch.uint8).cpu().numpy(), cdf_group_index)
        rv = torch.as_tensor(rv)
        return rv.to(device).to(dtype)
    
    def decode_z(self, total_size, cdf_group_index, start_offset, per_channel_size):
        self.decoder.decode_z(total_size, cdf_group_index, start_offset, per_channel_size)
        
    def get_decoded_tensor(self, device, dtype, non_blocking=False):
        rv = self.decoder.get_decoded_tensor()
        rv = torch.as_tensor(rv)
        return rv.to(device, non_blocking=non_blocking).to(dtype)
    
    def set_use_two_entropy_coders(self, use_two_entropy_coders):
        self.encoder.set_use_two_encoders(use_two_entropy_coders)
        self.decoder.set_use_two_decoders(use_two_entropy_coders)
        
        
class Bitparm(nn.Module):
    def __init__(self, qp_num, channel, final=False):
        super().__init__()
        self.final = final
        self.h = nn.Parameter(torch.nn.init.normal_(
            torch.empty([qp_num, channel, 1, 1]), 0, 0.01))                 # entropy_botleneck._matrix  (qp_num, channel, 1)
        self.b = nn.Parameter(torch.nn.init.normal_(
            torch.empty([qp_num, channel, 1, 1]), 0, 0.01))                 # entropy_botleneck._bias    (qp_num, channel, 1)
        if not final:
            self.a = nn.Parameter(torch.nn.init.normal_(
                torch.empty([qp_num, channel, 1, 1]), 0, 0.01))
        else:
            self.a = None

    def forward(self, x, index):
        h = torch.index_select(self.h, 0, index)
        b = torch.index_select(self.b, 0, index)
        x = x * F.softplus(h) + b
        if self.final:
            return x
        a = torch.index_select(self.a, 0, index)    # type: ignore 
        return x + torch.tanh(x) * torch.tanh(a)
    

class AEHelper(nn.Module):
    def __init__(self):
        super().__init__()
        self.entropy_coder = None
        self.cdf_group_index = None
        # self._offset = None
        # self._quantized_cdf = None
        # self._cdf_length = None
        self._offset = torch.IntTensor()
        self._quantized_cdf = torch.IntTensor()
        self._cdf_length = torch.IntTensor()

    def set_cdf_info(self, quantized_cdf, cdf_length, offset):
        self._quantized_cdf = quantized_cdf.cpu().numpy()
        self._cdf_length = cdf_length.reshape(-1).int().cpu().numpy()
        self._offset = offset.reshape(-1).int().cpu().numpy()

    def get_cdf_info(self):
        return self._quantized_cdf, \
            self._cdf_length, \
            self._offset

class BitEstimator(AEHelper):
    """BitEstimator for z (hyperprior) entropy modeling.
    
    Supports both training and inference modes:
    - Training: Use get_z_bits() to compute rate for backpropagation
    - Inference: Use update() to initialize entropy coder, then encode/decode
    """
    def __init__(self, qp_num, channel):
        super().__init__()
        self.f1 = Bitparm(qp_num, channel)
        self.f2 = Bitparm(qp_num, channel)
        self.f3 = Bitparm(qp_num, channel)
        self.f4 = Bitparm(qp_num, channel, True)
        self.qp_num = qp_num
        self.channel = channel

    def forward(self, x, index):
        """Forward pass, returns CDF values for input x.
        
        Args:
            x: Input tensor, shape (B, C, H, W)
            index: QP index tensor, shape (B,)
        Returns:
            CDF values, shape (B, C, H, W)
        """
        return self.get_cdf(x, index)

    def get_logits_cdf(self, x, index):
        """Get logits before sigmoid activation."""
        x = self.f1(x, index)
        x = self.f2(x, index)
        x = self.f3(x, index)
        x = self.f4(x, index)
        return x

    def get_cdf(self, x, index):
        """Get CDF values through sigmoid activation."""
        return torch.sigmoid(self.get_logits_cdf(x, index))   # follow Open DCVCs

    def get_z_bits(self, z, index):
        """Compute bits for z during training (supports backpropagation).
        Args:
            z: Quantized z values, shape (B, C, H, W)
            index: QP index tensor, shape (B,)
        Returns:
            bits: Bits per element, shape (B, C, H, W)
        """
        # # Compute probability: P(z - 0.5 < Z <= z + 0.5)
        # probs = self.get_cdf(z + 0.5, index) - self.get_cdf(z - 0.5, index)
        # # Clamp to avoid log(0)
        # probs = probs.clamp(min=1e-6)
        # # Convert to bits: -log2(prob)
        # bits = -torch.log(probs) / math.log(2.0)
        # return bits
        # NOTE: OpenDCVCs
        prob_clamp = 1e-6
        # Compute probability: P(z-0.5 < Z <= z+0.5)
        probs = self.get_cdf(z+0.5, index) - self.get_cdf(z-0.5, index)
        probs = probs.float()
        # Calculate safe log probability to avoid log(0) errors
        def safe_log_prob(p, eps=1e-10):
            # Direct log calculation when probability is above threshold
            # Otherwise use Laplacian approximation for numerical stability
            log_p = torch.where(
                p > prob_clamp,
                # Normal
                torch.log(torch.clamp(p, min=eps)),
                # Using properties of Laplace distribution for approximation
                # Laplace distribution has exponential decay in tails, so linear approximation works well
                # This is consistent with the BitEstimator model characteristics, one-order taylor expansion
                torch.log(torch.tensor(prob_clamp, device=p.device, dtype=p.dtype)) + (p - prob_clamp) / prob_clamp
            )
            return log_p
        log_prob = safe_log_prob(probs)
        bits = -log_prob / math.log(2.0)
        bits = torch.clamp(bits, min=0, max=50)
        return bits
        

    def update(self, force=True, entropy_coder=None):
        """Update CDF tables for entropy coding (inference mode).
        This method should be called before encode/decode operations.
        It pre-computes CDF tables for all channels and QP values.
        """
        
        # if self.entropy_coder is not None and not force:
        #     return
        
        self.entropy_coder = entropy_coder
        # self.cdf_helper = AEHelper()

        # 去掉梯度
        with torch.no_grad():
            device = next(self.parameters()).device
            medians = torch.zeros((self.qp_num, self.channel, 1, 1), device=device)
            index = torch.arange(self.qp_num, device=device, dtype=torch.int32)

            # 扩展搜索边界以覆盖较大的尾部概率，避免硬编码 8 导致大量旁路编码
            max_search = 50         # NOTE: origin 8
            minima = medians + max_search
            for i in range(max_search, 1, -1):
                samples = torch.zeros_like(medians) - i
                probs = self.forward(samples, index)
                minima = torch.where(probs < torch.zeros_like(medians) + 0.0001,
                                     torch.zeros_like(medians) + i, minima)

            maxima = medians + max_search
            for i in range(max_search, 1, -1):
                samples = torch.zeros_like(medians) + i
                probs = self.forward(samples, index)
                maxima = torch.where(probs > torch.zeros_like(medians) + 0.9999,
                                     torch.zeros_like(medians) + i, maxima)

            minima = minima.int()
            maxima = maxima.int()

            offset = -minima

            pmf_start = medians - minima
            pmf_length = maxima + minima + 1

            max_length = pmf_length.max()
            device = pmf_start.device
            samples = torch.arange(max_length, device=device) # pyright: ignore

            samples = samples[None, None, None, :] + pmf_start

            half = float(0.5)

            lower = self.forward(samples - half, index)
            upper = self.forward(samples + half, index)
            pmf = upper - lower

            pmf = pmf[:, :, 0, :]
            upper = self.forward(maxima.to(torch.float32), index)
            tail_mass = lower[:, :, 0, :1] + (1.0 - upper[:, :, 0, -1:])

            pmf = pmf.reshape([-1, max_length])
            tail_mass = tail_mass.reshape([-1, 1])
            pmf_length = pmf_length.reshape([-1])
            offset = offset.reshape([-1])
            quantized_cdf = EntropyCoder.pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
            cdf_length = pmf_length + 2
            self.set_cdf_info(quantized_cdf, cdf_length, offset)
            self.cdf_group_index = self.entropy_coder.add_cdf(*self.get_cdf_info()) # type: ignore

    def build_indexes(self, size, qp):
        B, C, H, W = size
        indexes = torch.arange(C, dtype=torch.int).view(1, -1, 1, 1) + qp * self.channel
        return indexes.repeat(B, 1, H, W)

    def encode_z(self, x, qp):
        _, _, H, W = x.size()
        return self.entropy_coder.encode_z(x.reshape(-1), self.cdf_group_index, qp * self.channel,
                                           H * W)

    def decode_z(self, size, qp):
        self.entropy_coder.decode_z(self.channel * size[0] * size[1], self.cdf_group_index,
                                    qp * self.channel, size[0] * size[1])

    def get_z(self, size, device, dtype):
        output_size = (1, self.channel, size[0], size[1])
        val = self.entropy_coder.get_decoded_tensor(device, dtype, non_blocking=True)
        return val.reshape(output_size)


class GaussianEncoder(AEHelper):
    """Gaussian/Laplacian entropy encoder for y (latent) entropy modeling.
    
    Supports both training and inference modes:
    - Training: Use get_y_bits() to compute rate for backpropagation
    - Inference: Use update() to initialize entropy coder, then encode/decode
    # From Balle's tensorflow compression examples
    SCALES_MIN = 0.11
    SCALES_MAX = 256
    SCALES_LEVELS = 64
    Args:
        distribution: 'gaussian' or 'laplace', default is 'gaussian'
    """
    def __init__(self, distribution='gaussian'):
        super().__init__()
        assert distribution in ['gaussian', 'laplace'], f"Unsupported distribution: {distribution}"
        self.distribution = distribution
        
        # Set scale parameters based on distribution
        if distribution == 'laplace':
            self.scale_min = 0.01
            self.scale_max = 64.0
            self.scale_level = 256
        else:  # gaussian
            self.scale_min = 0.11
            self.scale_max = 16.0          # 16.0  64.0
            self.scale_level = 128  # <= 256； 128
            
        scale_table = self.get_scale_table(self.scale_min, self.scale_max, self.scale_level)
        
        self.register_buffer('scale_table', scale_table)

        self.log_scale_min = math.log(self.scale_min)
        self.log_scale_max = math.log(self.scale_max)
        self.log_scale_step = (self.log_scale_max - self.log_scale_min) / (self.scale_level - 1)
        self.log_step_recip = 1. / self.log_scale_step

        self.decode_index_cache = {}
        self.decode_zeros_cache = {}

    @staticmethod
    def get_scale_table(min_val, max_val, levels):
        return torch.exp(torch.linspace(math.log(min_val), math.log(max_val), levels))

    def _get_distribution(self, loc, scale):
        """Get probability distribution instance."""
        if self.distribution == 'laplace':
            return torch.distributions.laplace.Laplace(loc, scale)
        else:  # gaussian
            return torch.distributions.normal.Normal(loc, scale)

    def get_y_bits(self, y, scales, means=None):
        """Compute bits for y during training (supports backpropagation).
        Args:
            y: Quantized y values, shape (B, C, H, W)
            scales: Scale parameters, shape (B, C, H, W)
            means: Mean parameters (optional), shape (B, C, H, W), default is 0
        Returns:
            bits: Bits per element, shape (B, C, H, W)
        """
        # if means is None:
        #     means = torch.zeros_like(y)
        # # Clamp scales to valid range
        # scales = scales.clamp(min=self.scale_min, max=1e10)
        # # Create distribution
        # distribution = self._get_distribution(means, scales)
        # # Compute probability: P(y - 0.5 < Y <= y + 0.5)
        # probs = distribution.cdf(y + 0.5) - distribution.cdf(y - 0.5)
        # # Clamp to avoid log(0)
        # probs = probs.clamp(min=1e-6)
        # # Convert to bits: -log2(prob)
        # bits = -torch.log(probs) / math.log(2.0)
        # return bits
        # NOTE: OpenDCVCs
        @torch.no_grad()  # We don't need gradient through the assert check
        def check_sigma(s):
            assert s.min() > 0, f"Invalid sigma value: {s.min()}"
            return s 
        y = y.float()
        # 
        scales = scales.clamp(min=self.scale_min, max=1e10)
        scales = check_sigma(scales.float())
        if means is None:
            means = torch.zeros_like(scales)
        distribution = self._get_distribution(means, scales)
        # probs = gaussian.cdf(y + 0.5) - gaussian.cdf(y - 0.5)
        # Safe log probability mass calculation
        def safe_log_prob_mass(dist, x, bin_size=1.0, prob_clamp=1e-6):
            # Calculate probability mass: CDF(x+0.5) - CDF(x-0.5)
            prob_mass = dist.cdf(x+0.5*bin_size) - dist.cdf(x-0.5*bin_size)
            # Use approximation for numerical stability when probability mass is small
            log_prob = torch.where(
                prob_mass > prob_clamp,
                torch.log(torch.clamp(prob_mass, min=1e-10)),
                # Use log of PDF times bin size as approximation
                # Normal.log_prob calculates -(x-mu)^2 / (2*sigma^2) analytically,
                # For Laplace distribution, this is a good approximation
                dist.log_prob(x) + math.log(bin_size)
            )
            return log_prob, prob_mass
        # Calculate log probability and probability mass
        log_probs, probs = safe_log_prob_mass(distribution, y)
        # Convert from nats to bits but DO NOT sum
        bits = torch.clamp(-log_probs / math.log(2.0), min=0, max=50.0)
        return bits
            

    def get_y_bits_with_quant_step(self, y, scales, means=None, quant_step=None):
        """Compute bits for y with quantization step during training.
        
        Args:
            y: y values before quantization, shape (B, C, H, W)
            scales: Scale parameters, shape (B, C, H, W)
            means: Mean parameters (optional), shape (B, C, H, W)
            quant_step: Quantization step (optional), shape (B, C, H, W) or scalar
            
        Returns:
            bits: Bits per element, shape (B, C, H, W)
            y_hat: Quantized y values, shape (B, C, H, W)
        """
        if means is None:
            means = torch.zeros_like(y)
        if quant_step is None:
            quant_step = torch.ones_like(y)
        
        # Apply quantization step
        y_q = y * quant_step
        means_q = means * quant_step
        scales_q = scales * quant_step
        
        # Clamp scales
        scales_q = scales_q.clamp(min=self.scale_min, max=1e10)
        
        # Round to get quantized values
        y_hat = torch.round(y_q)
        
        # Compute bits
        bits = self.get_y_bits(y_hat, scales_q, means_q)
        
        # Dequantize
        y_hat = y_hat / quant_step
        
        return bits, y_hat

    def update(self, force=True, entropy_coder=None, force_zero_thres=None):
        
        # if entropy_coder is not None:
        #     self.entropy_coder = entropy_coder
        # if not force:
        #     return
        
        # self.cdf_helper = AEHelper()
        self.entropy_coder = entropy_coder
        self.force_zero_thres = force_zero_thres

        # 动态计算最大搜索边界，确保覆盖 99.99% 的概率质量 NOTE: origin 8
        # Gaussian 分布 4.5 个 sigma 约覆盖 99.99%，Laplace 需要约 9 个 scale
        # max_search = int(self.scale_max * (10 if self.distribution == 'laplace' else 5)) + 1
        max_search = 50
        pmf_center = torch.zeros_like(self.scale_table) + max_search
        scales = torch.zeros_like(pmf_center) + self.scale_table
        cdf_distribution = self._get_distribution(torch.zeros_like(scales), scales)
        for i in range(max_search, 1, -1):
            samples = torch.zeros_like(pmf_center) + i
            probs = cdf_distribution.cdf(samples)
            probs = torch.squeeze(probs)
            pmf_center = torch.where(probs > torch.zeros_like(pmf_center) + 0.9999,
                                     torch.zeros_like(pmf_center) + i, pmf_center)

        pmf_center = pmf_center.int()
        pmf_length = 2 * pmf_center + 1
        max_length = torch.max(pmf_length).item()

        device = pmf_center.device
        samples = torch.arange(max_length, device=device) - pmf_center[:, None]
        samples = samples.float()

        scales = torch.zeros_like(samples) + self.scale_table[:, None]
        cdf_distribution = self._get_distribution(torch.zeros_like(scales), scales)

        upper = cdf_distribution.cdf(samples + 0.5)
        lower = cdf_distribution.cdf(samples - 0.5)
        pmf = upper - lower

        tail_mass = 2 * lower[:, :1]

        quantized_cdf = torch.Tensor(len(pmf_length), max_length + 2)
        quantized_cdf = EntropyCoder.pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)

        self.set_cdf_info(quantized_cdf, pmf_length+2, -pmf_center)
        self.cdf_group_index = self.entropy_coder.add_cdf(*self.get_cdf_info())


    def build_indexes_decoder(self, scales):
        scales = scales.reshape(-1)
        indexes, skip_cond = build_index_dec(scales, self.scale_min, self.scale_max,
                                             self.log_scale_min, self.log_step_recip,
                                             self.force_zero_thres)
        if self.force_zero_thres is not None:
            indexes = indexes[skip_cond]
        return indexes, skip_cond

    def build_indexes_encoder(self, symbols, scales):
        symbols = symbols.reshape(-1)
        scales = scales.reshape(-1)
        symbols = build_index_enc(symbols, scales, self.scale_min, self.scale_max,
                                  self.log_scale_min, self.log_step_recip, self.force_zero_thres)
        return symbols

    def encode_y(self, x, scales):
        symbols = self.build_indexes_encoder(x, scales)
        return self.entropy_coder.encode_y(symbols, self.cdf_group_index)

    def get_decode_index_cache(self, num, device):
        if num not in self.decode_index_cache:
            c = torch.arange(0, num, dtype=torch.int32, device=device)
            self.decode_index_cache[num] = c

        return self.decode_index_cache[num]

    def get_decode_zeros_cache(self, num, device):
        if num not in self.decode_zeros_cache:
            c = torch.zeros(num, dtype=torch.int32, device=device)
            self.decode_zeros_cache[num] = c

        return self.decode_zeros_cache[num].clone()

    def decode_and_get_y(self, scales, dtype, device):
        indexes, skip_cond = self.build_indexes_decoder(scales)
        self.decode_y(indexes)
        return self.get_y(scales.shape, scales.numel(), dtype, device, skip_cond, indexes)

    def decode_y(self, indexes):
        self.entropy_coder.decode_y(indexes, self.cdf_group_index)

    def get_y(self, shape, numel, dtype, device, skip_cond, indexes):
        if len(indexes) == 0:
            return torch.zeros(shape, dtype=dtype, device=device)
        if skip_cond is not None:
            curr_index = self.get_decode_index_cache(numel, device)
            back_index = self.get_decode_zeros_cache(numel, device)
            back_index.masked_scatter_(skip_cond, curr_index)
        val = self.entropy_coder.get_decoded_tensor(device, dtype, non_blocking=True)
        if skip_cond is not None:
            y = torch.index_select(val, 0, back_index) * skip_cond
            return y.reshape(shape)
        return val.reshape(shape)

