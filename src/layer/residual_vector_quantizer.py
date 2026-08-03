
from typing import List, Optional

import torch
from torch import nn
from .vector_quantizer import VectorQuantizerProj, VectorQuantizerIBQProj


# ---------------------------------------------------------------------------
# 梯度分数混合（参考 vector-quantize-pytorch residual_vq.py: frac_gradient）
# 控制残差量化链中较早层的梯度流量:
#   frac=0  → 完全截断（默认，与原行为一致）
#   frac=1  → 完整梯度流向前面的量化器
# ---------------------------------------------------------------------------
def frac_gradient(t: torch.Tensor, frac: float) -> torch.Tensor:
    """frac * t + (1 - frac) * t.detach()"""
    if frac <= 0.:
        return t.detach()
    return frac * t + (1. - frac) * t.detach()



class ResidualVectorQuantizer(nn.Module):
    """
    可训练的残差 VQ，带 quantizer dropout：
    - 每个 batch 随机采样使用的码本数 m ∈ {1, ..., n_codebooks}
    - 训练目标支持"对所有 m 的重建都负责"（与你之前讨论的公式 (8) 对齐）

    新增参数（参考 vector-quantize-pytorch ResidualVQ）:
        kmeans_init:              是否对各层码本使用 k-means 延迟初始化
        kmeans_iters:             k-means 迭代次数
        quant_grad_frac:          残差链中梯度流向前层比例: 0=完全截断, 1=完整梯度
        threshold_ema_dead_code:  死码复活阈值: cluster_size < 此值的码字将被替换。0=禁用
    """
    def __init__(
        self,
        n_codebooks: int,
        n_e: int,       # K
        in_dim: int,    # C
        codebook_dim: int = 8,
        beta: float = 0.25,
        use_ema: bool = True,
        decay: float = 0.99,
        eps: float = 1e-5,
        dropout_prob: float = 1.0,   # 1.0 represents always dropout; 0.0 represents always use all codebooks
        kmeans_init: bool = False,
        kmeans_iters: int = 10,
        quant_grad_frac: float = 0.,
        threshold_ema_dead_code: int = 2,
        quantizer_type: str = 'standard',     # IBQ, Opt
        use_double_quant: bool = True,
    ):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.dropout_prob = dropout_prob
        self.quant_grad_frac = quant_grad_frac
        
        if quantizer_type == 'standard':
            self.quantizers = nn.ModuleList([
                VectorQuantizerProj(
                    n_e, in_dim, codebook_dim=codebook_dim,
                    beta=beta, use_ema=use_ema, decay=decay, eps=eps,
                    kmeans_init=kmeans_init, kmeans_iters=kmeans_iters,
                    threshold_ema_dead_code=threshold_ema_dead_code,
                )
                for _ in range(n_codebooks)
            ])
        elif quantizer_type == 'IBQ':
            self.quantizers = nn.ModuleList([
                VectorQuantizerIBQProj(
                    n_e, in_dim, codebook_dim=codebook_dim,
                    beta=beta, kmeans_init=kmeans_init, kmeans_iters=kmeans_iters,
                    use_double_quant=use_double_quant,
                )
                for _ in range(n_codebooks)
            ])

    # ---------- 训练前向 ----------
    def forward(
        self,
        z: torch.Tensor,
        return_all_levels: bool = False,
    ):
        """
        z: [B, C, H, W]
        return_all_levels: True 时返回每个 m 对应的 z_q_m，用于多粒度损失
        返回字典：
            z_q:        [B, C, H, W]     cumulative reconstruction
            indices:    [n_codebooks, B, H, W]   quantized indices
            vq_loss:    scalar           sum of VQ losses
            z_q_levels: List[Tensor]     cumulative reconstructions for each m, None if not return_all_levels
        """
        B = z.size(0)

        residual = z
        z_q_cum  = torch.zeros_like(z)
        vq_loss  = z.new_zeros(())    # 0
        perplexity = z.new_zeros(())    # 0
        indices: List[torch.Tensor] = []
        levels:  List[torch.Tensor] = []

        # 2) 残差量化主循环
        for i, q in enumerate(self.quantizers):
            out = q(residual)
            z_q_i, loss_i, ind_i = out["z_q_out"], out["vq_loss"], out["indices"]

            # cumulative add
            z_q_cum = z_q_cum + z_q_i
            vq_loss = vq_loss + loss_i
            perplexity = perplexity + out["perplexity"]
            indices.append(ind_i)
            if return_all_levels:
                levels.append(z_q_cum.clone())
            # Residual update – 通过 frac_gradient 控制梯度流向较早层
            # quant_grad_frac=0 → 完全截断（等价于 .detach()）; =1 → 完整梯度
            residual = residual - frac_gradient(z_q_i, self.quant_grad_frac)

        # STE: z_q_cum 通过底层的 VectorQuantizerProj.forward 已做 STE，
        # 此处仅聚合各层输出。如需对累积输出再做额外 STE，取消下行注释：
        # z_q_cum = z + (z_q_cum - z).detach()

        packed = torch.stack(indices, dim=0)  # [n_codebooks, B, H, W]
        avg_vq_loss = vq_loss / self.n_codebooks           # average loss over all codebooks

        return {
            "z_q": z_q_cum,
            "indices": packed,
            "vq_loss": avg_vq_loss,
            "perplexity": perplexity,
            "z_q_levels": levels if return_all_levels else None,
        }

    # ---------- inference cache initialization ----------
    @torch.inference_mode()
    def init_inference_cache(self):
        """推理前需为所有子量化器初始化缓存"""
        for q in self.quantizers:
            q.init_inference_cache()
        return self

    # ---------- inference encoding ----------
    @torch.inference_mode()
    def encoding(self, z: torch.Tensor) -> torch.Tensor:
        residual = z.clone()
        indices = []
        for q in self.quantizers:
            ind_i = q.encode_index(residual)
            indices.append(ind_i)
            z_q_i = q.decoding(ind_i)
            residual = residual - z_q_i
        return torch.stack(indices, dim=0)

    @torch.inference_mode()
    def decoding(self, inds: torch.Tensor) -> torch.Tensor:
        n = inds.shape[0]
        z_q = self.quantizers[0].decoding(inds[0])
        for i in range(1, n):
            z_q = z_q + self.quantizers[i].decoding(inds[i])
        return z_q
