import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .wnconv import WNConv2d
from .ef_lic_layer import FastEmbedding
import math


# ---------------------------------------------------------------------------
# K-Means helper (adapted from vector-quantize-pytorch)
# ---------------------------------------------------------------------------
def kmeans_init_(
    samples: torch.Tensor,          # [N, D]
    num_clusters: int,
    num_iters: int = 10,
    use_cosine_sim: bool = False,
):
    """Run batched-less k-means on flat samples, return (centroids, cluster_size)."""
    N, D = samples.shape
    device, dtype = samples.device, samples.dtype

    # 1) randomly sample initial centroids
    if N >= num_clusters:
        indices = torch.randperm(N, device=device)[:num_clusters]
    else:
        indices = torch.randint(0, N, (num_clusters,), device=device)
    means = samples[indices].clone()  # [K, D]
    bins = torch.zeros(num_clusters, device=device, dtype=torch.long)

    for _ in range(num_iters):
        # 2) compute distances
        if use_cosine_sim:
            # cosine similarity as distance
            samples_norm = F.normalize(samples, p=2, dim=-1)
            means_norm = F.normalize(means, p=2, dim=-1)
            dists = samples_norm @ means_norm.t()  # [N, K], higher = better
        else:
            # euclidean distance (negative, so argmax still works)
            x2 = (samples ** 2).sum(dim=-1, keepdim=True)   # [N, 1]
            y2 = (means ** 2).sum(dim=-1)                    # [K]
            xy = samples @ means.t()                         # [N, K]
            dists = -(x2 + y2.unsqueeze(0) - 2.0 * xy)      # [N, K]

        # 3) assign to nearest centroid
        buckets = dists.argmax(dim=-1)  # [N]

        # 4) compute new means
        bins = torch.zeros(num_clusters, device=device, dtype=torch.long)
        bins.scatter_add_(0, buckets, torch.ones_like(buckets, dtype=torch.long))
        zero_mask = bins == 0
        bins_clamped = bins.masked_fill(zero_mask, 1).to(dtype)

        new_means = torch.zeros(num_clusters, D, device=device, dtype=dtype)
        new_means.scatter_add_(0, buckets.unsqueeze(-1).expand(-1, D), samples)
        new_means = new_means / bins_clamped.unsqueeze(-1)

        if use_cosine_sim:
            new_means = F.normalize(new_means, p=2, dim=-1)

        # 5) keep old means where a cluster died
        means = torch.where(zero_mask.unsqueeze(-1), means, new_means)

    return means, bins


class VectorQuantizerProj(nn.Module):
    """
    可训练 + 可推理的向量量化模块。
    训练时走 forward()，推理时先调用 init_inference_cache() 后走 encode_index / encoding / decoding。
    n_e: codebook size,
    in_dim: input dimension,
    codebook_dim: codebook dimension,

    新增参数:
        kmeans_init:              是否使用 k-means 初始化码本（延迟初始化，在首次 forward 时执行）
        kmeans_iters:             k-means 迭代次数
        threshold_ema_dead_code:  死码复活阈值：cluster_size < 此值的码字将被随机替换。
                                  0 表示禁用。参考 vector-quantize-pytorch 默认值 2.
        reset_cluster_size:       复活码字时重置的 cluster_size，默认 = threshold_ema_dead_code
    """
    def __init__(
        self,
        n_e: int,
        in_dim: int,
        codebook_dim: int,
        beta: float = 0.25,
        use_ema: bool = True,
        decay: float = 0.99,
        eps: float = 1e-5,
        kmeans_init: bool = False,
        kmeans_iters: int = 10,
        threshold_ema_dead_code: int = 0,
        reset_cluster_size: Optional[int] = None,
    ):
        super().__init__()
        self.n_e = n_e              # K
        self.e_dim = codebook_dim   # D
        self.beta = beta
        self.use_ema = use_ema
        self.decay = decay
        self.eps = eps
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.has_dead_code_replacement = threshold_ema_dead_code > 0
        self.reset_cluster_size = reset_cluster_size if reset_cluster_size is not None else threshold_ema_dead_code

        self.embedding = FastEmbedding(n_e, codebook_dim)

        if kmeans_init:
            # 延迟初始化: 在首次 forward 时用 k-means 结果填充
            nn.init.zeros_(self.embedding.weight)
        else:
            nn.init.uniform_(self.embedding.weight, -1.0 / n_e, 1.0 / n_e)
        # nn.init.normal_(self.embedding.weight, std=1.0 / math.sqrt(codebook_dim))

        self.in_proj  = WNConv2d(in_dim, codebook_dim)
        self.out_proj = WNConv2d(codebook_dim, in_dim)


        self.register_buffer("counter", torch.zeros(n_e))
        # Always register cluster_size for dead code tracking (EMA or gradient mode)
        self.register_buffer("cluster_size", torch.zeros(n_e))
        if use_ema:
            self.register_buffer("ema_embed", self.embedding.weight.data.clone())
            self.embedding.weight.requires_grad_(False)
        else:
            # Gradient-based codebook: embedding is trainable, initialize with uniform
            # (k-means will overwrite this on first forward if kmeans_init=True)
            pass

        # k-means 延迟初始化标记
        self.register_buffer("_kmeans_initted", torch.tensor(not kmeans_init))

        # 推理缓存占位
        self._codebook_T = None
        self._codebook_norm_sq = None
        self._decode_codebook = None
        self._decode_dim = None

    # ---------------------------------------------------------------
    # Dead code revival (参考 vector-quantize-pytorch Codebook.replace / expire_codes_)
    # ---------------------------------------------------------------
    @staticmethod
    def _sample_vectors(samples: torch.Tensor, num: int) -> torch.Tensor:
        """从 [N, D] 中随机采样 num 个向量，样本不足时有放回。"""
        num_samples, device = samples.shape[0], samples.device
        if num_samples >= num:
            indices = torch.randperm(num_samples, device=device)[:num]
        else:
            indices = torch.randint(0, num_samples, (num,), device=device)
        return samples[indices]

    @torch.no_grad()
    def replace_dead_codes_(self, batch_samples: torch.Tensor):
        """
        将 cluster_size 低于阈值的死码替换为 batch 中的随机样本，
        同时重置对应 cluster_size 和 ema_embed，给予复活码字公平的竞争机会。

        参考 vector-quantize-pytorch Codebook.replace
        """
        dead_mask = self.cluster_size < self.threshold_ema_dead_code  # [K]
        if not dead_mask.any():
            return

        num_dead = dead_mask.sum().item()
        replacements = self._sample_vectors(batch_samples, num_dead)  # [num_dead, D]

        self.embedding.weight.data[dead_mask] = replacements.to(self.embedding.weight.dtype)
        self.cluster_size.data[dead_mask] = self.reset_cluster_size
        if self.use_ema:
            self.ema_embed.data[dead_mask] = replacements * self.reset_cluster_size

    @torch.no_grad()
    def expire_codes_(self, batch_samples: torch.Tensor):
        """训练时检测并复活死码。threshold_ema_dead_code <= 0 时为空操作。"""
        if not self.has_dead_code_replacement or not self.training:
            return
        self.replace_dead_codes_(batch_samples)

    # ---------------------------------------------------------------
    # K-Means lazy initialisation (参考 vector-quantize-pytorch Codebook.init_embed_)
    # ---------------------------------------------------------------
    @torch.no_grad()
    def _try_kmeans_init(self, z_flat: torch.Tensor):
        """若 kmeans_init=True 且尚未初始化，则对 z_flat 执行 k-means 并填充码本与 EMA buffer。"""
        if self._kmeans_initted:
            return

        embed, cluster_size = kmeans_init_(
            z_flat,
            self.n_e,
            num_iters=self.kmeans_iters,
            use_cosine_sim=False,
        )  # embed: [K, D], cluster_size: [K]

        # Always save cluster sizes for dead code tracking
        self.cluster_size.data.copy_(cluster_size)

        if self.use_ema:
            # Initialize EMA buffers consistent with update_ema format
            self.ema_embed.data.copy_(embed * cluster_size.unsqueeze(-1))
            # Laplace smoothing for codebook update
            n = self.cluster_size.sum()
            cs = (self.cluster_size + self.eps) / (n + self.n_e * self.eps) * n
            self.embedding.weight.data.copy_(self.ema_embed / cs.unsqueeze(-1))
        else:
            self.embedding.weight.data.copy_(embed)

        self._kmeans_initted.data.copy_(torch.tensor(True))

    # ---------------------------------------------------------------
    # For training    ✔️
    # ---------------------------------------------------------------
    def _distances(self, z_flat: torch.Tensor) -> torch.Tensor:
        """entire distance matrix。z_flat: [N,D] return [N,K]。"""
        emb = self.embedding.weight                              # [K, D]
        z_sq = z_flat.pow(2).sum(dim=1, keepdim=True)            # [N, 1]
        e_sq = emb.pow(2).sum(dim=1)                             # [K]
        # d_{ij} = ||z_i||^2 + ||e_j||^2 - 2 z_i·e_j^T
        d = z_sq + e_sq.unsqueeze(0) - 2.0 * z_flat @ emb.t()    # [N, K]
        return d

    @torch.no_grad()
    def find_nearest(self, z_flat: torch.Tensor) -> torch.Tensor:
        d = self._distances(z_flat)  # [N, K]
        return d.argmin(dim=1)       # indices [N,]

    def get_quantized(self, indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(indices)

    def forward(self, z: torch.Tensor):
        """
        z: [B, in_dim, H, W]
        return:
            z_q_out      : [B, in_dim, H, W]   (pass through out_proj)
            vq_loss      : scalar              (commitment + codebook)
            indices      : [B, H, W]           (long)
            perplexity   : scalar              (codebook usage monitoring)
        """
        # 1) Project to codebook dimension
        z = self.in_proj(z)                                      # [B, D, H, W]
        B, D, H, W = z.shape

        # 2) Flatten token sequence
        z_perm = z.permute(0, 2, 3, 1).contiguous()              # [B, H, W, D]
        z_flat = z_perm.view(-1, D)                              # [N, D]

        # 3) K-means 延迟初始化 (参考 vector-quantize-pytorch Codebook.init_embed_)
        self._try_kmeans_init(z_flat)

        # 4) Nearest neighbor lookup
        indices = self.find_nearest(z_flat)

        # 5) take codebook lookup
        z_q_flat = self.get_quantized(indices)                      # [N, D]

        # 6) loss
        vq_loss = self.compute_loss(z_flat, z_q_flat)

        # 7) STE
        z_q_flat = z_flat + (z_q_flat - z_flat).detach()
        z_q = z_q_flat.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()        # [B, H, W, D] -> [B, D, H, W]

        # 8) Codebook update
        if self.training:
            if self.use_ema:
                self.update_ema(z_flat, indices)
            else:
                # Gradient-based mode: track cluster sizes via EMA for dead code detection
                # (codebook weights are updated by optimizer gradients, not EMA)
                one_hot = F.one_hot(indices, self.n_e).to(z_flat.dtype)
                batch_cluster_size = one_hot.sum(dim=0)
                self.cluster_size.data.mul_(self.decay).add_(batch_cluster_size, alpha=1 - self.decay)
            self.expire_codes_(z_flat)

        # 9) monitoring
        with torch.no_grad():
            avg_probs  = F.one_hot(indices, self.n_e).float().mean(dim=0)   # [N, K] -> [K]
            perplexity = torch.exp(-(avg_probs * (avg_probs + 1e-10).log()).sum())

        # 10) Project to input dimension
        z_q_out = self.out_proj(z_q)                              # [B, in_dim, H, W]

        return {
            "z_q_out": z_q_out,
            "vq_loss": vq_loss,
            "indices": indices.view(B, H, W),
            "perplexity": perplexity,
        }

    @torch.no_grad()
    def update_ema(self, z_flat: torch.Tensor, indices: torch.Tensor):
        one_hot = F.one_hot(indices, self.n_e).to(z_flat.dtype)         # [N, K] 
        cluster_size = one_hot.sum(dim=0)                               # [K]      nt
        embed_sum = one_hot.t() @ z_flat                                # [K, N] @ [N, D] -> [K, D], st
        # ema update
        self.cluster_size.mul_(self.decay).add_(cluster_size, alpha=1-self.decay)
        self.ema_embed.mul_(self.decay).add_(embed_sum, alpha=1-self.decay)
        
        # laplace smoothing
        n = self.cluster_size.sum()
        cs = (self.cluster_size + self.eps) / (n + self.n_e * self.eps) * n 
        #  update codebook
        self.embedding.weight.data.copy_(self.ema_embed / cs.unsqueeze(1))  # [K, D] / [K, 1]
        
        # monitoring 
        self.counter.scatter_add_(0, indices, torch.ones_like(indices, dtype=self.counter.dtype))

    def compute_loss(self, z_flat: torch.Tensor, z_q_flat: torch.Tensor) -> torch.Tensor:
        
        commitment_loss = F.mse_loss(z_q_flat.detach(), z_flat)
        
        if self.use_ema:
            return self.beta * commitment_loss
        
        codebook_loss = F.mse_loss(z_q_flat, z_flat.detach())
        return codebook_loss + self.beta * commitment_loss 

    # ---------------------------------------------------------------
    # For inference
    # ---------------------------------------------------------------
    @torch.inference_mode()
    def init_inference_cache(self):
        emb = self.embedding.weight.detach()
        self._codebook_T       = emb.t().contiguous()
        self._codebook_norm_sq = emb.square().sum(dim=1).contiguous()
        w = self.out_proj.weight().flatten(1)                     # [C_in, D]
        self._decode_codebook  = emb.matmul(w.t())
        self._decode_codebook.add_(self.out_proj.bias)
        self._decode_codebook  = self._decode_codebook.contiguous()
        self._decode_dim       = self._decode_codebook.shape[1]
        return self

    def _nearest_infer(self, z_flat: torch.Tensor) -> torch.Tensor:
        d = torch.mm(z_flat, self._codebook_T)
        d.mul_(-2.0).add_(self._codebook_norm_sq)
        return d.argmin(1)

    def _decode_flat(self, indices, out: Optional[torch.Tensor] = None):
        idx = indices.to(device=self._decode_codebook.device,
                         dtype=torch.long, non_blocking=True).reshape(-1)
        if out is None:
            return torch.index_select(self._decode_codebook, 0, idx)
        torch.index_select(self._decode_codebook, 0, idx, out=out)
        return out

    @torch.inference_mode()
    def encode_index(self, z):
        z = self.in_proj(z)
        B, C, H, W = z.shape
        ind = self._nearest_infer(z.permute(0, 2, 3, 1).reshape(-1, C))
        return ind.view(B, H, W)

    @torch.inference_mode()
    def encoding(self, z, flat_buffer: Optional[torch.Tensor] = None):
        z = self.in_proj(z)
        B, C, H, W = z.shape
        ind = self._nearest_infer(z.permute(0, 2, 3, 1).reshape(-1, C))
        flat = self._decode_flat(ind, out=flat_buffer)
        return ind.view(B, H, W), flat.view(B, H, W, self._decode_dim).permute(0, 3, 1, 2)

    @torch.inference_mode()
    def decoding(self, indices: torch.Tensor):
        B, H, W = indices.shape
        flat = self._decode_flat(indices)
        return flat.view(B, H, W, self._decode_dim).permute(0, 3, 1, 2).contiguous()



# # ============================================================================
# # [NEW] IBQProj — Index Backpropagation Quantization
# # ============================================================================
# # Reference:
# #   Fengyuan Shi, Zhuoyan Luo, Yixiao Ge, Yujiu Yang, Ying Shan, Limin Wang.
# #   "Scalable Image Tokenization with Index Backpropagation Quantization."
# #   arXiv:2412.02692v2 (2024), CVPR 2025.
# #
# # 算法核心（Algorithm 1 + Eq. 6~7）:
# #     logits      = z @ C^T                          # [N, K]
# #     Ind_soft    = softmax(logits, dim=1)           # [N, K]
# #     Ind_hard    = one_hot(argmax(Ind_soft))        # [N, K]
# #     Ind         = Ind_hard - Ind_soft.detach() + Ind_soft   ← STE on categorical dist.
# #     z_q         = Ind @ C                          # [N, D]
# #
# # 与 VQGAN 的关键区别：
# #   VQGAN 在 *选中的那条码字* 上做 STE —— 每次只有 K/N 个码字更新，其它码字逐渐 stale，分布 gap 拉大。
# #   IBQ   在 *全部 K 条码字的概率分布* 上做 STE —— 所有码字按 softmax 概率得到梯度，
# #         编码器与码本的分布在训练中保持一致。
# #
# # 训练损失（Eq. 12，论文称为 *double quantization loss*）:
# #     z_q' = Ind_hard^T @ C                         # 纯 hard 量化（无 STE）
# #     L_Q  = ||z_q - z||^2                          ← (a) 对齐损失，梯度同时回 encoder & codebook
# #         + ||sg[z] - z_q'||^2                      ← (b) 码本损失，梯度只回 codebook
# #         + beta * ||z - sg[z_q']||^2               ← (c) commitment，梯度只回 encoder
# #
# class VectorQuantizerIBQProj(nn.Module):
#     """
#     Index Backpropagation Quantization (IBQ)
#     — 见 Shi et al., "Scalable Image Tokenization with Index Backpropagation Quantization" (CVPR 2025)。

#     forward 签名与 VectorQuantizerProj 完全一致：
#         x : [B, in_dim, H, W]
#         return : {"z_q_out", "vq_loss", "indices", "perplexity"}

#     训练完成后调用 ``init_inference_cache()`` 即可使用与 VectorQuantizerProj 一致的
#     encode_index / encoding / decoding 推理接口。

#     Args:
#         n_e               : 码本大小 K。
#         in_dim            : 编码/解码通道维度。
#         codebook_dim      : 码字维度 D。
#         beta              : commitment 损失权重（论文实验通常 0.25）。
#         kmeans_init       : 是否用 K-means 延迟初始化码本（强烈推荐 True）。
#         kmeans_iters      : K-means 迭代次数。
#         temperature       : softmax 温度（论文用 1.0；调小可让分布更尖）。
#         use_double_quant  : 是否启用论文 Eq. 12 的全部三项损失
#                             （True = 与论文一致；False = 只用 (a) 对齐项，更轻量）。
#         init_std          : 当 kmeans_init=False 时，码本初始化的 Gaussian 标准差系数。

#     说明：本模块与 VQGAN 的 EMA / 死码复活机制**不兼容**，因为 EMA 只更新部分码字，
#     会破坏 IBQ 的「all-codes 更新」特性。即便极度小概率出现极少量死码，
#     论文实验表明 96% 的利用率已足够，无需额外干预。
#     """

#     def __init__(
#         self,
#         n_e: int,    # K
#         in_dim: int,
#         codebook_dim: int,   # D
#         beta: float = 0.25,
#         kmeans_init: bool = True,
#         kmeans_iters: int = 10,
#         temperature: float = 1.0,
#         use_double_quant: bool = True,
#         init_std: float = 1.0,
#     ):
#         super().__init__()
#         assert n_e > 0 and codebook_dim > 0
#         self.n_e            = n_e                  # K
#         self.e_dim          = codebook_dim         # D
#         self.beta           = beta
#         self.kmeans_init    = kmeans_init
#         self.kmeans_iters   = kmeans_iters
#         self.temperature    = float(temperature)
#         self.use_double_quant = use_double_quant

#         # ------------------------------------------------------------------
#         # Codebook —— 用 FastEmbedding（与原代码一致）
#         # ------------------------------------------------------------------
#         self.embedding = FastEmbedding(n_e, codebook_dim)
#         if kmeans_init:
#             # 延迟初始化：首次 forward 时用 K-means 填充
#             nn.init.zeros_(self.embedding.weight)
#         else:
#             nn.init.normal_(self.embedding.weight,
#                             std=init_std / math.sqrt(codebook_dim))
#         # IBQ 没有 EMA，码本始终是 gradient-trained
#         self.embedding.weight.requires_grad_(True)

#         # ------------------------------------------------------------------
#         # Projections —— 与原代码共享同一套 WNConv2d 风格
#         # ------------------------------------------------------------------
#         self.in_proj  = WNConv2d(in_dim, codebook_dim)
#         self.out_proj = WNConv2d(codebook_dim, in_dim)

#         # ------------------------------------------------------------------
#         # 状态 buffer
#         # ------------------------------------------------------------------
#         # K-means 延迟初始化标志
#         self.register_buffer("_kmeans_initted", torch.tensor(not kmeans_init))
#         # 推理缓存占位
#         self._codebook_T       = None
#         self._decode_codebook  = None
#         self._decode_dim       = None

#     # ----------------------------------------------------------------
#     # K-Means lazy initialisation（与 VectorQuantizerProj 完全一致的策略）
#     # ----------------------------------------------------------------
#     @torch.no_grad()
#     def _try_kmeans_init(self, z_flat: torch.Tensor):
#         """若 kmeans_init=True 且尚未初始化，则对 z_flat 执行 K-means 并填充码本。"""
#         if self._kmeans_initted:
#             return

#         embed, cluster_size = kmeans_init_(
#             z_flat,
#             self.n_e,
#             num_iters=self.kmeans_iters,
#             use_cosine_sim=False,
#         )  # embed: [K, D], cluster_size: [K]

#         # K-means 的 cluster_size 仅用于初始化（IBQ 没有死码复活）
#         self.embedding.weight.data.copy_(embed.to(self.embedding.weight.dtype))
#         self._kmeans_initted.data.copy_(torch.tensor(True))

#     # ----------------------------------------------------------------
#     # IBQ 量化核心 —— Algorithm 1 + Eq. 6~7
#     # ----------------------------------------------------------------
#     def quantize(self, z_flat: torch.Tensor):
#         """
#         把 [N, D] 的连续特征通过 IBQ 量化为离散 token。

#         Returns
#         -------
#         z_q   : [N, D]  IBQ STE 量化结果（差异于 z；梯度同时回 encoder 与 codebook）
#         z_q_h : [N, D]  纯 hard 量化结果（用于 loss 中作为 stationary target）
#         ind   : [N]     hard 索引
#         """
#         codebook = self.embedding.weight                            # [K, D]

#         # 1) z 与 C 的点积作为全部 K 个码字的 logits
#         logits = z_flat @ codebook.t()                              # [N, K]
#         # logits = (z_flat.pow(2).sum(1, keepdim=True) 
#         #             - 2 * z_flat @ codebook.t() 
#         #             + codebook.pow(2).sum(1))
        
#         if self.temperature != 1.0:
#             logits = logits / self.temperature

#         # 2) soft categorical distribution（用作梯度路由的"桥梁"）
#         ind_soft = F.softmax(logits, dim=-1)                        # [N, K]

#         # 3) hard index（前向用它；loss 里用作不可微目标）
#         ind = ind_soft.argmax(dim=-1)                               # [N]
#         ind_hard = F.one_hot(ind, self.n_e).to(z_flat.dtype)       # [N, K]

#         # 4) STE on categorical distribution —— *IBQ 的灵魂*
#         #    前向 = hard；反向 = soft（即按 p_k 全码字等比例回传梯度）
#         ind_ste = ind_hard - ind_soft.detach() + ind_soft           # [N, K]

#         # 5) 量化
#         z_q   = ind_ste  @ codebook                                 # [N, D]，IBQ STE 输出
#         z_q_h = ind_hard @ codebook                                 # [N, D]，纯 hard 输出
#         return z_q, z_q_h, ind

#     # ----------------------------------------------------------------
#     # IBQ 量化损失 —— Eq. 12 (double quantization loss)
#     # ----------------------------------------------------------------
#     def compute_loss(
#         self,
#         z_flat: torch.Tensor,           # [N, D]  编码器输出
#         z_q:    torch.Tensor,           # [N, D]  IBQ STE 输出（含梯度）
#         z_q_h:  torch.Tensor,           # [N, D]  hard 量化输出（含梯度回 codebook）
#     ) -> torch.Tensor:
#         """
#         论文 Eq. 12:
#             z_q' = Ind_hard^T @ C
#             L_Q  = ||z_q   - z||^2                  ← (a) IBQ 对齐
#                 + ||sg[z] - z_q'||^2                ← (b) 码本损失
#                 + beta * ||z - sg[z_q']||^2         ← (c) commitment

#         梯度流向：
#             (a) 同时回 encoder 和 codebook —— 这是 IBQ 与 VQGAN 最本质的差异。
#                 因为 z_q = Ind_ste @ C, ∂L/∂Ind_ste 经 softmax 同时影响 z 与 C。
#             (b) sg[z] 把 z 的梯度截断，所以只有 codebook 更新。
#             (c) sg[z_q'] 把 z_q_h 的梯度截断，所以只有 encoder (z) 更新。

#         当 use_double_quant=False 时，只保留 (a) 项 —— 编码器与码本通过 IBQ STE 自动对齐，
#         不需要额外的 codebook/commitment 项也能训练（论文 Tab. 2 显示单独 (a) 即可达到
#         1.67 rFID / 98% usage; 加上 (b)+(c) 后到 1.55 rFID / 97%）。
#         """
#         # (a) IBQ 对齐损失 —— 双向梯度
#         align_loss = F.mse_loss(z_q, z_flat)

#         if not self.use_double_quant:
#             return align_loss

#         # (b) 码本损失 —— 码本被拉向编码器特征
#         codebook_loss = F.mse_loss(z_q_h, z_flat.detach())

#         # (c) Commitment —— 编码器被拉向 hard 量化码字
#         commitment_loss = self.beta * F.mse_loss(z_q_h.detach(), z_flat)

#         return align_loss + codebook_loss + commitment_loss

#     # ----------------------------------------------------------------
#     # Forward —— 与 VectorQuantizerProj 同一签名 / 同一返回 dict
#     # ----------------------------------------------------------------
#     def forward(self, z: torch.Tensor):
#         """
#         z : [B, in_dim, H, W]
#         Returns
#         -------
#         z_q_out    : [B, in_dim, H, W]  反卷积投影后的特征（送入 decoder）
#         vq_loss    : scalar              L_Q —— Eq. 12 的 IBQ 损失
#         indices    : [B, H, W]           离散 token (long)
#         perplexity : scalar              码本利用率监控（论文报告：~96%）
#         """
#         # 1) Project 到 codebook 维度
#         z = self.in_proj(z)                                              # [B, D, H, W]
#         B, D, H, W = z.shape

#         # 2) 拉平成 token 序列
#         z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, D)          # [N, D]

#         # 3) K-means 延迟初始化
#         self._try_kmeans_init(z_flat)

#         # 4) IBQ 量化
#         z_q, z_q_h, ind = self.quantize(z_flat)                          # [N,D], [N,D], [N]

#         # 5) 量化损失 L_Q
#         vq_loss = self.compute_loss(z_flat, z_q, z_q_h)

#         # 6) reshape 回 feature map
#         z_q = z_q.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()       # [B, D, H, W]
#         # z_q_h 留作 loss 目标，不再往下游传

#         # 7) 监控 perplexity（码本利用率的常用代理指标）
#         with torch.no_grad():
#             avg_probs  = F.one_hot(ind, self.n_e).float().mean(dim=0)    # [K]
#             perplexity = torch.exp(-(avg_probs * (avg_probs + 1e-10).log()).sum())

#         # 8) 投影回输入维度
#         z_q_out = self.out_proj(z_q)                                       # [B, in_dim, H, W]

#         return {
#             "z_q_out":   z_q_out,
#             "vq_loss":   vq_loss,
#             "indices":   ind.view(B, H, W),
#             "perplexity": perplexity,
#         }

#     # ========================================================================
#     # Inference cache / encode / decode  —— 与 VectorQuantizerProj 接口一致
#     # ========================================================================

#     @torch.inference_mode()
#     def init_inference_cache(self):
#         """推理前调用一次，把码本 & decode 表预处理到 buffer。"""
#         emb = self.embedding.weight.detach()                                # [K, D]
#         self._codebook_T = emb.t().contiguous()                            # [D, K]

#         # 预折叠 out_proj，得到"索引 → 输出特征"的一步直查表
#         w = self.out_proj.weight().flatten(1)                              # [C_in, D]
#         self._decode_codebook = emb.matmul(w.t())                          # [K, C_in]
#         self._decode_codebook.add_(self.out_proj.bias)                     # broadcast
#         self._decode_codebook = self._decode_codebook.contiguous()
#         self._decode_dim = self._decode_codebook.shape[1]
#         return self

#     @torch.inference_mode()
#     def _nearest_infer(self, z_flat: torch.Tensor) -> torch.Tensor:
#         """IBQ 推理时用 logits argmax 选码字（论文 §3.1：`equivalent to argmax of softmax`）。"""
#         logits = torch.mm(z_flat, self._codebook_T)                         # [N, K]
#         if self.temperature != 1.0:
#             logits = logits / self.temperature
#         return logits.argmax(dim=-1)                                        # [N]

#     def _decode_flat(self, indices, out: Optional[torch.Tensor] = None):
#         idx = indices.to(device=self._decode_codebook.device,
#                          dtype=torch.long, non_blocking=True).reshape(-1)
#         if out is None:
#             return torch.index_select(self._decode_codebook, 0, idx)
#         torch.index_select(self._decode_codebook, 0, idx, out=out)
#         return out

#     @torch.inference_mode()
#     def encode_index(self, z):
#         """
#         z : [B, in_dim, H, W]
#         return : indices [B, H, W] (long)
#         """
#         z = self.in_proj(z)
#         B, C, H, W = z.shape
#         ind = self._nearest_infer(z.permute(0, 2, 3, 1).reshape(-1, C))
#         return ind.view(B, H, W)

#     @torch.inference_mode()
#     def encoding(self, z, flat_buffer: Optional[torch.Tensor] = None):
#         """
#         一次完成 in-proj + 选 index + 出 out_proj 后的特征。
#         """
#         z = self.in_proj(z)
#         B, C, H, W = z.shape
#         ind = self._nearest_infer(z.permute(0, 2, 3, 1).reshape(-1, C))
#         flat = self._decode_flat(ind, out=flat_buffer)
#         return ind.view(B, H, W), flat.view(B, H, W, self._decode_dim).permute(0, 3, 1, 2)

#     @torch.inference_mode()
#     def decoding(self, indices: torch.Tensor):
#         """
#         indices : [B, H, W] (long)
#         return  : [B, in_dim, H, W]
#         """
#         B, H, W = indices.shape
#         flat = self._decode_flat(indices)
#         return flat.view(B, H, W, self._decode_dim).permute(0, 3, 1, 2).contiguous()
    

def compute_entropy_loss(
    logits: torch.Tensor, temperature: float = 0.01, sample_minimization_weight: float=1.0,
    batch_maximization_weight: float=1.0, eps: float=1e-5
):
    """
    Entropy loss of unnormalized logits

    logits: Affinities are over the last dimension

    https://github.com/google-research/magvit/blob/05e8cfd6559c47955793d70602d62a2f9b0bdef5/videogvt/train_lib/losses.py#L279
    LANGUAGE MODEL BEATS DIFFUSION — TOKENIZER IS KEY TO VISUAL GENERATION (2024)
    """
    # logits [N, K]
    probs = F.softmax(logits / temperature, dim=-1)   # [N, K]
    log_probs = F.log_softmax(logits / temperature + eps, dim=-1)
    
    avg_probs = probs.mean(dim=tuple(range(probs.ndim - 1)))    # mean expect last dim
    avg_entropy = -torch.sum(avg_probs * torch.log(avg_probs + eps))
    
    sample_entropy = -torch.sum(probs * log_probs, dim=-1)
    sample_entropy = sample_entropy.mean()
    
    loss = (sample_minimization_weight * sample_entropy) - (batch_maximization_weight * avg_entropy)
    
    return sample_entropy, avg_entropy, loss 

    
class VectorQuantizerIBQProj(VectorQuantizerProj):
    def __init__(
        self,
        n_e: int,
        in_dim: int,
        codebook_dim: int,
        beta: float = 0.25,
        use_ema: bool = True,
        decay: float = 0.99,
        eps: float = 1e-5,
        kmeans_init: bool = False,
        kmeans_iters: int = 10,
        threshold_ema_dead_code: int = 0,
        reset_cluster_size: Optional[int] = None,
        use_entropy_loss: bool = False,
        entropy_temperature: float = 0.01,
        sample_minimization_weight: float=1.0,
        batch_maximization: float=1.0,
        use_double_quant: bool = True,
        entropy_loss_weight: float = 0.01,
                 ):
        super().__init__(
            n_e=n_e,
            in_dim=in_dim,
            codebook_dim=codebook_dim,
            beta=beta,
            use_ema=use_ema,
            decay=decay,
            eps=eps,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            threshold_ema_dead_code=threshold_ema_dead_code,
            reset_cluster_size=reset_cluster_size,
        )
        
        self.use_entropy_loss = use_entropy_loss
        self.entropy_temperature = entropy_temperature
        self.sample_minimization_weight = sample_minimization_weight
        self.batch_maximization = batch_maximization
        self.use_double_quant = use_double_quant
        self.entropy_loss_weight = entropy_loss_weight
        
    
    def forward(self, z: torch.Tensor):
        """
        z: [B, in_dim, H, W]
        """
        
        z = self.in_proj(z)  # [B, D, H, W]
        B, D, H, W = z.size()
        
        z_perm = z.permute(0, 2, 3, 1).contiguous()     # [B, H, W, D]
        z_flat = z_perm.view(-1, D)                     # [B*H*W, D]
        
        self._try_kmeans_init(z_flat)
        
        # IBQ inference 
        z_q, z_q_h, logits, indices = self.index_backprop_quant(z_flat)
        
        # compute loss 
        quant_loss = self.compute_loss(z_flat, z_q, z_q_h)
        
        if self.use_entropy_loss:
            sample_entropy, avg_entropy, entropy_loss = compute_entropy_loss(
                logits, temperature=self.entropy_temperature, sample_minimization_weight=self.sample_minimization_weight,
                batch_maximization_weight=self.batch_maximization, eps=self.eps
            )
            quant_loss = quant_loss + entropy_loss * self.entropy_loss_weight 
        
        
        z_q = z_q.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()       # [B, D, H, W]
        
        z_q_out = self.out_proj(z_q)
        
        with torch.no_grad():
            avg_probs  = F.one_hot(indices, self.n_e).float().mean(dim=0)    # [K]
            perplexity = torch.exp(-(avg_probs * (avg_probs + 1e-10).log()).sum())
            
        return {
            "z_q_out":   z_q_out,
            "vq_loss":   quant_loss,
            "indices":   indices.view(B, H, W),
            "perplexity": perplexity,
        }
        
    
    def index_backprop_quant(self, z_flat: torch.Tensor):
        # z_flat: [N, D]
        codebook = self.embedding.weight                # [K, D]
        logits = z_flat @ codebook.t()     # [N, K]
        
        ind_soft = F.softmax(logits, dim=1)         # [N, K]
        _, indices = ind_soft.max(dim=1, keepdim=True)                # [N]
        
        # ind_hard = F.one_hot(indices, num_classes=self.n_e)     # [N, K]
        ind_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(1, indices, 1.0)
        ind = ind_hard - ind_soft.detach() + ind_soft           # for gradient
        
        z_q = ind @ codebook
        
        z_q_h = ind_hard @ codebook
        
        return z_q, z_q_h, logits, indices
    
    
    def compute_loss(self, z_flat: torch.Tensor, z_q: torch.Tensor, z_q_h: torch.Tensor):  # type: ignore
        
        loss = F.mse_loss(z_flat, z_q)
        
        if self.use_double_quant:
            comment_loss = F.mse_loss(z_flat, z_q_h.detach())
            codebook_loss = F.mse_loss(z_flat.detach(), z_q_h)
            
            loss = loss + comment_loss * self.beta + codebook_loss 
        
        return loss 
    
        
        
