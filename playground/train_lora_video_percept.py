
# -*- encoding: utf-8 -*-

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.dataset.vimeo90k_dataset import Vimeo90kDataset
from src.metrics.metric import calculate_metrics
# from src.models.image_model import DCVCRTImage
# from src.models.video_model import DCVCRTVideo
from src.models.lora_image_model import DCVCRTImage
from src.models.lora_video_model import DCVCRTVideo
from src.utils.options import load_yaml_config, DotDict
from src.utils.transforms import ycbcr2rgb
from src.utils.utils import AverageMeter, get_state_dict, set_seed

from src.discriminator.hific_discriminator import HiFiCConditionalDiscriminator
from src.losses.perceptual_loss import LPIPSLoss
from src.losses.gan_loss import VanillaGANLoss
import src.loralib as loralib

QP = None

# =========================================================================
#  Distributed helpers
# =========================================================================
def setup_distributed():
    """初始化分布式训练环境。单 GPU 时返回 (0, 1, 0)。"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        backend = "gloo" if sys.platform == "win32" else "nccl"
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def get_raw_model(model):
    """DDP 包装时返回内部模型，否则返回自身。"""
    return model.module if isinstance(model, DistributedDataParallel) else model


def reduce_dict(metrics, device):
    """跨所有 GPU 平均指标字典。"""
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return metrics
    world_size = dist.get_world_size()
    reduced = {}
    for k, v in metrics.items():
        t = torch.tensor(v, device=device, dtype=torch.float32)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        reduced[k] = t.item() / world_size
    return reduced


# =========================================================================
#  Logger setup — 训练日志 & 验证日志分开
# =========================================================================
def setup_logger(save_dir, is_main=True):
    """分别创建 train / val 两个 logger，仅主进程写文件。"""
    train_logger = logging.getLogger("train_logger")
    train_logger.setLevel(logging.INFO)
    train_logger.propagate = False
    if train_logger.hasHandlers():
        train_logger.handlers.clear()

    val_logger = logging.getLogger("val_logger")
    val_logger.setLevel(logging.INFO)
    val_logger.propagate = False
    if val_logger.hasHandlers():
        val_logger.handlers.clear()

    if is_main:
        fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        train_logger.addHandler(ch)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fh_train = logging.FileHandler(os.path.join(save_dir, 'train.log'))
            fh_train.setLevel(logging.INFO)
            fh_train.setFormatter(fmt)
            train_logger.addHandler(fh_train)

            fh_val = logging.FileHandler(os.path.join(save_dir, 'val.log'))
            fh_val.setLevel(logging.INFO)
            fh_val.setFormatter(fmt)
            val_logger.addHandler(fh_val)

    return train_logger, val_logger


train_logger = logging.getLogger("train_logger")
val_logger = logging.getLogger("val_logger")


# =========================================================================
#  Criterion — 返回 per-sample (B,) 值，由调用方 .mean()
# =========================================================================
# class VideoCriterion(nn.Module):
#     """视频帧级率失真损失，返回 per-sample 值。"""

#     def __init__(self, lambdas=None):
#         super().__init__()
#         self.mse = nn.MSELoss(reduction="none")
#         if lambdas is None:
#             l_min, l_max = 1, 768
#             qt = np.linspace(0, 63, 64)
#             lambdas = np.exp(np.log(l_min) + qt / (64 - 1) * (np.log(l_max) - np.log(l_min)))
#         self.register_buffer("lambdas", torch.tensor(lambdas, dtype=torch.float32))

#     def forward(self, output, target, qp_index, weight=1.0):
#         x_hat = output["x_hat"]
#         bpp_raw = output["bpp"]

#         rgb_x_hat, rgb_x = ycbcr2rgb(x_hat), ycbcr2rgb(target)
        
#         ycbcr_mse = self.mse(x_hat, target).mean(dim=(1, 2, 3))
        
#         # temp = mse_per_pixel.mean(dim=[2, 3])
#         # ycbcr_mse = (temp[:, 0] * 4 + temp[:, 1] + temp[:, 2]) / 6.0

#         rgb_mse = self.mse(rgb_x_hat, rgb_x).mean(dim=(1, 2, 3))

#         k = 0.8
#         mse = k * ycbcr_mse + (1 - k) * rgb_mse

#         loss = self.lambdas[qp_index.long()] * mse * weight + bpp_raw     # type: ignore 

#         return {"bpp": bpp_raw, "mse": mse, "loss": loss}


class RDPCriterion(nn.Module):
    def __init__(self, opt, lambdas=None):
        super().__init__()
        self.opt = opt 
        self.mse = nn.MSELoss(reduction='none')
        self.gan_loss = VanillaGANLoss(**opt.gan_loss)
        self.perceptual_loss = LPIPSLoss(**opt.perceptual_loss)
        self.percept_weight = opt.perceptual_loss_weight 
        self.gen_weight = opt.gen_loss_weight
        self.distortion_weight = opt.distortion_loss_weight
        
    def get_distortion(self, x_ycbcr_hat, x_ycbcr, x_rgb_hat, x_rgb):
        yuv420_mse = self.mse(x_ycbcr_hat, x_ycbcr).mean(dim=[1, 2, 3])
        rgb_mse = self.mse(x_rgb_hat, x_rgb).mean(dim=[1, 2, 3])
        k = 0.8
        return (k * yuv420_mse + (1 - k) * rgb_mse) * self.distortion_weight
    
    def get_perceptual_loss(self, x_rgb_hat, x_rgb):
        return self.perceptual_loss(x_rgb_hat, x_rgb) * self.percept_weight    
    
    def get_adv_loss(self, d_fake):
        return self.gan_loss(d_fake, is_real=True, is_disc=False) * self.gen_weight 
    
    def compression_loss(self, x_ycbcr_hat, x_ycbcr, x_rgb_hat, x_rgb, d_fake):
        # Optimize Generator 
        res = {}

        res["mse_loss"] = self.get_distortion(x_ycbcr_hat, x_ycbcr, x_rgb_hat, x_rgb)
        
        res["perceptual_loss"] = self.get_perceptual_loss(x_rgb_hat, x_rgb).reshape(x_rgb_hat.shape[0], -1).mean(dim=1)
        
        res["adv_loss"] = self.get_adv_loss(d_fake)
        
        res["loss"] = res["mse_loss"] + res["perceptual_loss"] + res["adv_loss"]
        # (B, )
        return res 
        
    
    def disc_loss(self, d_real, d_fake):
        # Optimize Discriminator
        res = {}
        
        res["d_real_loss"] = self.gan_loss(d_real, is_real=True, is_disc=True) * 0.5
        res["d_fake_loss"] = self.gan_loss(d_fake, is_real=False, is_disc=True)  * 0.5 
        
        res["disc_loss"] = res["d_real_loss"] + res["d_fake_loss"]
        # (B, )
        return res 
    

# =========================================================================
#  Helpers
# =========================================================================

def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False


def _get_frame_offset(total_frames, pnum, mode="random", rng=None):
    """在 [0, T-1-pnum] 范围内返回起始帧偏移。"""
    max_offset = total_frames - 1 - pnum
    if max_offset <= 0:
        return 0
    if mode == "random":
        if rng is not None:
            return rng.randint(0, max_offset + 1)
        return np.random.randint(0, max_offset + 1)
    if mode == "center":
        return max_offset // 2
    return 0


# =========================================================================
#  Model Wrapper — Intra (frozen) + Inter (trainable)
# =========================================================================

class TradVideoModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.intra_model = self._load_intra_model()
        self.intra_model.eval()
        self.intra_model.update()
        freeze_model(self.intra_model)

        self.video_model = DCVCRTVideo(z_channel=config.inter_model.z_channel)
        if getattr(config.inter_model, "ckpt", None) is not None:
            self.video_model.load_state_dict(get_state_dict(config.inter_model.ckpt), strict=False)
            print(f"[InterModel] Loaded from {config.inter_model.ckpt}")
        else:
            print("[InterModel] Warning: random init (no pretrained ckpt).")

        loralib.mark_only_lora_as_trainable(self.video_model, bias="lora_only")
        
        self.weights = [0.5, 1.2, 0.5, 0.9, 0.5, 0.9, 0.5, 0.9]
        self.offsets = [0, 8, 0, 4, 0, 4, 0, 4]
        self.rng = np.random.RandomState(20260320)
        self.current_stage = 0

    def _load_intra_model(self):
        model = DCVCRTImage(N=self.config.intra_model.N,
                            z_channel=self.config.intra_model.z_channel)
        ckpt = getattr(self.config.intra_model, "ckpt", None)
        lora_ckpt = getattr(self.config.intra_model, "lora_ckpt", None)
        if ckpt and os.path.exists(ckpt):
            model.load_state_dict(get_state_dict(ckpt), strict=False)
            model.load_state_dict(get_state_dict(lora_ckpt), strict=False)
            print(f"[IntraModel] Loaded from {ckpt} and {lora_ckpt}")
        else:
            print("[IntraModel] Warning: random init (no pretrained ckpt).")
        return model

    @staticmethod
    def get_training_strategy(stage):
        """
        2 -> 129, 
        """
        if stage == 0:
            strategy = \
                [[0, 1e-4, 2, 256, 256]] * 5 + \
                [[5, 1e-4, 3, 256, 256]] * 5 + \
                [[10, 1e-4, 6, 256, 256]] * 45 + \
                [[55, 1e-4, 7, 256, 256]]
                # epoch, lr, pnum, h, w
        elif stage == 1:
            strategy = \
                [[0, 5e-5, 8, 256, 256]] * 5 + \
                [[5, 5e-5, 16, 256, 256]] * 5 + \
                [[10, 5e-5, 24, 256, 256]] * 5 + \
                [[15, 5e-5, 32, 256, 256]] * 15 + \
                [[30, 5e-6, 32, 256, 256]] * 7 + \
                [[37, 5e-6, 32, 256, 256]] 
        elif stage == 2:
            strategy = \
                [[0,   5e-5, 33,  512, 512]] * 14 + \
                [[14,  5e-6, 33,  512, 512]] * 4 + \
                [[18,  2e-5, 49,  512, 512]] * 7 + \
                [[25,  2e-6, 49,  512, 512]] * 2 + \
                [[27,  5e-6, 65,  512, 512]] * 7 + \
                [[34,  2e-6, 65,  512, 512]] * 6 + \
                [[40,  2e-6, 65,  512, 512]]  # noqa: E501 E221
        elif stage == 3:
            strategy = \
                [[0,   2e-6, 97,  512, 512]] * 2 + \
                [[2,   5e-7, 129,  512, 512]] * 2 + \
                [[4,   5e-7, 129,  512, 512]]  # noqa: E501 E221
        elif stage == 5:
            strategy = \
                [[0,   1e-4, 7,  512, 512]] * 10 + \
                [[10,   5e-5, 7,  512, 512]] * 10 + \
                [[20,   5e-6, 7,  512, 512]]  # noqa: E501 E221
        else:
            assert False, f"stage {stage} is not support."
    
        return strategy
    
    # @staticmethod
    # def get_stage(epoch):
    #     """
    #     1-5:  adaptor_i 
    #     """
    #     stages = [5, 10, 30, 50, 100]
    #     for i, v in enumerate(stages):
    #         if epoch <= v:
    #             return i + 1
    #     return -1

    # @staticmethod
    # def get_pnum(stage, total_frames=7):
    #     max_p = total_frames - 1
    #     if stage <= 2:
    #         return min(1, max_p)
    #     elif stage == 3:
    #         return min(2, max_p)
    #     elif stage == 4:
    #         return min(6, max_p)
    #     else:
    #         return min(32, max_p)

    def update(self):
        self.video_model.update()


# =========================================================================
#  Save / Load — 模型与训练状态分开，仅主进程执行
# =========================================================================
def save_model(model, config, filename='latest_video_model.pth.tar'):
    if not is_main_process():
        return
    save_dir = config.save_dir
    os.makedirs(save_dir, exist_ok=True)
    raw = get_raw_model(model)
    # raw.video_model.update()
    # torch.save(raw.video_model.state_dict(), os.path.join(save_dir, filename))
    torch.save(raw.state_dict(), os.path.join(save_dir, filename))
    train_logger.info('Model saved to: ' + os.path.join(save_dir, filename))

def save_lora_checkpoint(model, config, bias="lora_only", filename='latest_lora_checkpoint.pth.tar'):
    if not is_main_process():
        return 
    save_dir = config.save_dir 
    os.makedirs(save_dir, exist_ok=True)
    lora_checkpoint = loralib.lora_state_dict(model, bias=bias)
    torch.save(lora_checkpoint, os.path.join(save_dir, filename))
    train_logger.info('Lora checkpoint saved to: ' + os.path.join(save_dir, filename))


def save_training_state(optimizer, d_optimizer, warmup_scheduler, reduce_scheduler,
                        epoch, global_step, best_val_loss, history, config,
                        filename='latest_training_state.pth.tar'):
    if not is_main_process():
        return
    save_dir = config.save_dir
    os.makedirs(save_dir, exist_ok=True)
    state = {
        'epoch': epoch, 'global_step': global_step,
        'best_val_loss': best_val_loss,
        'optimizer_state_dict': optimizer.state_dict(),
        # 'patience_counter': patience_counter,
        # 'warmup_scheduler_state_dict': warmup_scheduler.state_dict(),
        # 'reduce_scheduler_state_dict': reduce_scheduler.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict(),
        'history': history,
    }
    torch.save(state, os.path.join(save_dir, filename))
    train_logger.info('Training state saved to: ' + os.path.join(save_dir, filename))


def save_checkpoint(model, discriminator, optimizer, d_optimizer, warmup_scheduler, reduce_scheduler,
                    epoch, global_step, best_val_loss, history, config,
                    is_best=False):
    # save_model(model, config, 'latest_model.pth.tar')
    save_lora_checkpoint(model, config, bias="lora_only", filename="latest_lora_checkpoint.pth.tar")
    save_model(discriminator, config, filename="latest_discriminator_model.pth.tar")
    save_training_state(optimizer, d_optimizer, warmup_scheduler, reduce_scheduler,
                        epoch, global_step, best_val_loss, history, config,
                        'latest_training_state.pth.tar')
    if is_best:
        # save_model(model, config, 'best_model.pth.tar')
        save_lora_checkpoint(model, config, bias="lora_only", filename="best_lora_checkpoint.pth.tar")
        train_logger.info('Best model saved.')


# =========================================================================
#  configure_main_optimizer — 两参数组
# =========================================================================
def configure_main_optimizer(model, inter_lr):
    """DDP 兼容：通过 get_raw_model 获取底层 video_model 的参数。"""
    raw = get_raw_model(model)
    # amort_params = [p for n, p in raw.video_model.named_parameters()
    #                 if "bit_estimator_z" not in n and "gaussian_encoder" not in n
    #                 and p.requires_grad]
    # hyper_params = [p for n, p in raw.video_model.named_parameters()
    #                 if ("bit_estimator_z" in n or "gaussian_encoder" in n)
    #                 and p.requires_grad]

    # inter = set(amort_params) & set(hyper_params)
    # union = set(amort_params) | set(hyper_params)
    # assert len(inter) == 0 and len(union) == len([p for p in raw.video_model.parameters() if p.requires_grad])
    
    # return optim.AdamW([
    #     {"params": amort_params, "lr": inter_lr},
    #     {"params": hyper_params, "lr": inter_lr},
    # ])
    return optim.AdamW([p for p in raw.parameters() if p.requires_grad], lr=inter_lr)


def configure_disc_optimizer(model, lr):
    raw = get_raw_model(model)
    return optim.AdamW([p for p in raw.parameters() if p.requires_grad], lr=lr)


# =========================================================================
#  train_one_epoch
# =========================================================================
def train_one_epoch(model, discriminator, criterion, optimizer, d_optimizer, train_loader, val_loader,
                    device, config, epoch, scheduler=None, step_interval=None,
                    global_step=0, val_multi_qp_interval=None):
    # model.train()
    raw = get_raw_model(model)
    raw_disc = get_raw_model(discriminator)
    raw.video_model.train()
    raw_disc.train()
    

    per_qp_meters = {
        qp: {'loss': AverageMeter(), 'mse': AverageMeter(), 'bpp': AverageMeter(),  
             'lpips': AverageMeter(), 'adv': AverageMeter(), 'disc': AverageMeter()}
        for qp in range(64)
    }
    epoch_loss = AverageMeter()
    epoch_bpp = AverageMeter()
    epoch_mse = AverageMeter()
    epoch_lpips = AverageMeter()
    epoch_adv = AverageMeter()
    epoch_disc = AverageMeter()

    val_metrics_list = []
    val_multi_qp_list = []

    # ---- Get stage ----
    stage = config.stage 
    strategy = raw.get_training_strategy(stage)   # epoch, lr, pnum, h, w
    idx = min(len(strategy)-1, epoch)
    _, lr, seq_len, patch_height, patch_width = strategy[idx]
    
    # set learning rate 
    for g in optimizer.param_groups:
        g['lr'] = lr 
    
    train_loader.dataset.set_patch_size((patch_height, patch_width))
    train_loader.dataset.set_frame_num(seq_len)
    # stage = raw.get_stage(epoch)
    # pnum = raw.get_pnum(stage, config.dataset.train_dataset.seq_len)
    pnum = seq_len - 1
    raw.current_stage = stage

    if is_main_process():
        pbar = tqdm(train_loader, desc=f'Epoch {epoch:03d} Train', ncols=150)
    else:
        pbar = train_loader

    for batch_idx, frames in enumerate(pbar):
        B, T_orig, C, H, W = frames.shape
        # frames = frames.to(device)
        # optimizer.zero_grad()
        # qp_index = torch.randint(0, 64, (B,), device=device)
        # qp_val = np.random.randint(0, 64)
        qp_val = QP if QP is not None else np.random.randint(0, 64)
        qp_index = torch.full((B,), qp_val, dtype=torch.long, device=device)

        # if global_step < 100:
        #     cur_pnum = T_orig - 1
        #     cur_stage = 0
        #     offset_mode = "start"
        # else:
        cur_pnum = pnum
        cur_stage = stage
        offset_mode = "random"
        start = _get_frame_offset(T_orig, cur_pnum, mode=offset_mode, rng=raw.rng)
        frames = frames[:, start:start + cur_pnum + 1]
        _, T, _, _, _ = frames.shape

        # ---- I-frame (Intra, no grad) ----
        raw.video_model.clear_dpb()
        # raw.video_model.set_curr_poc(0)

        with torch.inference_mode():
            ref_img = frames[:, 0].to(device)
            intra_out = raw.intra_model(ref_img, qp_index)
            raw.video_model.add_ref_frame(feature=None, frame=intra_out["x_hat"].detach())
        
        # ---- P-frame loop ----
        total_loss = torch.zeros(B, device=device)
        total_bpp = torch.zeros(B, device=device)
        total_mse = torch.zeros(B, device=device)
        total_lpips = torch.zeros(B, device=device)
        total_adv = torch.zeros(B, device=device)
        total_disc = torch.zeros(B, device=device)
        num_p = 0

        for t in range(1, T):
            p_frame = frames[:, t].to(device)
            
            cur_qp = qp_index + raw.offsets[t % 8]
            # out_net = raw.video_model(p_frame, cur_qp)
            # loss_dict = criterion(out_net, p_frame, qp_index, weight=raw.weights[t % 8])

            # --------------------- Optimize Generator -------------------------
            raw_disc.requires_grad_(False)
            raw.video_model.requires_grad_(True)
            optimizer.zero_grad()

            # forward 
            out_net = raw.video_model(p_frame, cur_qp)
            
            # color space convertion
            x_ycbcr = p_frame 
            x_ycbcr_hat = out_net['x_hat']
            x_rgb = ycbcr2rgb(x_ycbcr, clamp=False)
            x_rgb_hat = ycbcr2rgb(x_ycbcr_hat, clamp=False)
            
            d_fake = raw_disc(x_rgb_hat, out_net['y_hat'])
            loss_dict = criterion.compression_loss(x_ycbcr_hat, x_ycbcr, x_rgb_hat, x_rgb, d_fake)
            
            
            loss_dict['loss'].mean().backward()
            if getattr(config.train, 'clip_max_norm', 1.0):
                torch.nn.utils.clip_grad_norm_(raw.video_model.parameters(), getattr(config.train, 'clip_max_norm', 1.0))
            optimizer.step()
            
            # --------------------- Optimize Discriminator ---------------------
            raw_disc.requires_grad_(True)
            raw.video_model.requires_grad_(False)
            d_optimizer.zero_grad()
            
            # forward 
            d_real = raw_disc(x_rgb, out_net['y_hat'].detach())
            d_fake = raw_disc(x_rgb_hat.detach(), out_net['y_hat'].detach())
            
            d_out_criterion = criterion.disc_loss(d_real, d_fake)
            
            d_loss = d_out_criterion['disc_loss'].mean()
            
            d_loss.backward()
            nn.utils.clip_grad_norm_(discriminator.parameters(), getattr(config.train, 'clip_max_norm', 1.0))
            d_optimizer.step()
            

            # log train process 
            bpp = out_net["bpp"]
            mse = loss_dict["mse_loss"]
            percept = loss_dict['perceptual_loss']
            adv_l = loss_dict['adv_loss']
            disc_l = d_out_criterion['disc_loss']

            # if cur_stage == 1:
            #     total_loss += mse
            # else:
            total_loss += loss_dict["loss"].detach()
            

            total_bpp += bpp.detach()
            total_mse += mse.detach()
            total_lpips += percept.detach()
            total_adv += adv_l.detach()
            total_disc += disc_l.detach()
            num_p += 1

        if num_p > 0:
            avg_loss = (total_loss / num_p).mean()
            avg_bpp = (total_bpp / num_p).mean()
            avg_mse = (total_mse / num_p).mean()
            avg_lpips = (total_lpips / num_p).mean()
            avg_adv = (total_adv / num_p).mean()
            avg_disc = (total_disc / num_p).mean()
        else:
            # avg_loss = torch.tensor(0.0, device=device, requires_grad=True)
            # avg_bpp = torch.tensor(0.0, device=device)
            # avg_mse = torch.tensor(0.0, device=device)
            raise ValueError("num_p == 0")

        # avg_loss.backward()


        lr = optimizer.param_groups[0]["lr"]
        global_step += 1

        epoch_loss.update(avg_loss.item())
        epoch_bpp.update(avg_bpp.item())
        epoch_mse.update(avg_mse.item())
        epoch_lpips.update(avg_lpips.item())
        epoch_adv.update(avg_adv.item())
        epoch_disc.update(avg_disc.item())

        # ---- per-QP tracking ----
        for qp in torch.unique(qp_index, sorted=True):
            qp_val = qp.item()
            mask = (qp_index == qp)
            n_qp = mask.sum().item()
            per_qp_meters[qp_val]['loss'].update(
                (total_loss[mask] / max(num_p, 1)).mean().item(), n_qp)
            per_qp_meters[qp_val]['bpp'].update(
                (total_bpp[mask] / max(num_p, 1)).mean().item(), n_qp)
            per_qp_meters[qp_val]['mse'].update(
                (total_mse[mask] / max(num_p, 1)).mean().item(), n_qp)
            per_qp_meters[qp_val]['lpips'].update(
                (total_lpips[mask] / max(num_p, 1)).mean().item(), n_qp)
            per_qp_meters[qp_val]['adv'].update(
                (total_adv[mask] / max(num_p, 1)).mean().item(), n_qp)
            per_qp_meters[qp_val]['disc'].update(
                (total_disc[mask] / max(num_p, 1)).mean().item(), n_qp)

        # ---- step-based scheduler ----
        # if scheduler is not None and not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        #     if global_step % step_interval == 0:
        #         scheduler.step()

        # ---- per-QP train log flush every 10000 iters ----
        if global_step % 10000 == 0 and is_main_process():
            for qp in range(64):
                if per_qp_meters[qp]['loss'].count > 0:
                    train_logger.info(
                        f"[TRAIN_QP] {global_step} {qp} "
                        f"{per_qp_meters[qp]['loss'].avg:.6f} "
                        f"{per_qp_meters[qp]['mse'].avg:.6f} "
                        f"{per_qp_meters[qp]['bpp'].avg:.6f} "
                        f"{per_qp_meters[qp]['lpips'].avg:.6f} "
                        f"{per_qp_meters[qp]['adv'].avg:.6f} "
                        f"{per_qp_meters[qp]['disc'].avg:.6f} "
                    )
            per_qp_meters = {
                qp: {'loss': AverageMeter(), 'mse': AverageMeter(), 'bpp': AverageMeter(),
                     'lpips': AverageMeter(), 'adv': AverageMeter(), 'disc': AverageMeter()}
                for qp in range(64)
            }

        # ---- QP=32 val every 5000 iters ----
        skip_qp32 = (val_multi_qp_interval is not None
                     and global_step % val_multi_qp_interval == 0)
        if global_step % 1000 == 0 and not skip_qp32:
            vm = validate_qp32(model, val_loader, criterion, device, config,
                               epoch, global_step)
            
            # -------------- scheduler --------------------
            # if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            #     scheduler.step(vm['loss'])
            
            val_metrics_list.append(vm)

        # ---- multi-QP val ----
        if val_multi_qp_interval is not None and global_step % val_multi_qp_interval == 0:
            mqp = validate_multi_qp(model, val_loader, criterion, device, config,
                                    epoch, global_step)
            val_multi_qp_list.append({'step': global_step, 'results': mqp})
            save_lora_checkpoint(raw.video_model, config, bias="lora_only", filename="middle_iter_lora.pth.tar")

        if is_main_process():
            pbar.set_postfix({
                'loss': f'{avg_loss.item():.4f}',
                'bpp': f'{avg_bpp.item():.4f}',
                'lpips': f'{avg_lpips.item():.4f}',
                'mse': f'{avg_mse.item():.6f}',
                'adv': f'{avg_adv.item():.6f}',
                'disc': f'{avg_disc.item():.4f}',
                'lr': f'{lr:.6f}',
                'stage': f'{cur_stage}',
                "iter": f"{global_step}"
            })

    # ---- reduce epoch metrics across GPUs ----
    metrics = {'loss': epoch_loss.avg, 'bpp': epoch_bpp.avg, 'mse': epoch_mse.avg, 
               'lpips': epoch_lpips.avg, 'adv': epoch_adv.avg, 'disc': epoch_disc.avg}
    metrics = reduce_dict(metrics, device)

    return (metrics, global_step, val_metrics_list, val_multi_qp_list)


# =========================================================================
#  validate_qp32
# =========================================================================

@torch.inference_mode()
def validate_qp32(model, val_loader, criterion, device, config, epoch, global_step):
    # model.eval()
    raw = get_raw_model(model)
    raw.video_model.eval()

    loss_meter = AverageMeter()
    bpp_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    lpips_meter = AverageMeter()
    mse_meter = AverageMeter()

    # stage = raw.get_stage(epoch)
    # pnum = raw.get_pnum(stage, config.dataset.train_dataset.seq_len)
    # TODO: set frame number
    stage = config.stage
    strategy = raw.get_training_strategy(stage)
    idx = min(len(strategy)-1, epoch)
    _, lr, seq_len, patch_height, patch_width= strategy[idx]
    
    pnum = seq_len - 1
    val_loader.dataset.set_patch_size((patch_height, patch_width))
    val_loader.dataset.set_frame_num(seq_len)

    if is_main_process():
        pbar = tqdm(val_loader, desc=f'Val QP32 step {global_step}', ncols=150)
    else:
        pbar = val_loader

    for frames in pbar:
        B, T_orig, C, H, W = frames.shape
        # frames = frames.to(device)

        start = _get_frame_offset(T_orig, pnum, mode="center")
        frames = frames[:, start:start + pnum + 1]
        _, T, _, _, _ = frames.shape
        
        qp_val = QP if QP is not None else 32 
        qp_index = torch.full((B,), qp_val, dtype=torch.long, device=device)

        raw.video_model.clear_dpb()
        # raw.video_model.set_curr_poc(0)
        
        ref_frame = frames[:, 0].to(device)
        intra_out = raw.intra_model(ref_frame, qp_index)
        raw.video_model.add_ref_frame(feature=None, frame=intra_out["x_hat"].detach())

        total_loss = 0.0
        total_bpp = 0.0
        psnr_sum = 0.0
        ssim_sum = 0.0
        total_mse = 0.0
        lpips_v_sum = 0.0
        num_p = 0

        for t in range(1, T):
            
            cur_frame = frames[:, t].to(device)
            cur_qp = qp_index + raw.offsets[t % 8]
            
            out_net = raw.video_model(cur_frame, cur_qp)
            
            
            # color space conversion
            x_ycbcr = cur_frame 
            x_ycbcr_hat = out_net['x_hat']
            x_rgb = ycbcr2rgb(x_ycbcr)
            x_rgb_hat = ycbcr2rgb(x_ycbcr_hat)
            
            psnr_v, ssim_v = calculate_metrics(x_rgb_hat, x_rgb)
            
            # loss_dict = criterion(out_net, cur_frame, qp_index, weight=raw.weights[t % 8])
            lpips_v = criterion.get_perceptual_loss(x_rgb_hat, x_rgb)
            mse_loss = criterion.get_distortion(x_ycbcr_hat, x_ycbcr, x_rgb_hat, x_rgb)
            # adv_loss = criterion.get_adv_loss(x_rgb_hat, x_rgb)
            loss = mse_loss + lpips_v
            
            total_loss += loss.mean().item()
            total_bpp += out_net["bpp"].mean().item()
            psnr_sum += psnr_v
            ssim_sum += ssim_v
            lpips_v_sum += lpips_v.mean().item()
            total_mse += mse_loss.mean().item()
            num_p += 1

        n = max(num_p, 1)
        loss_meter.update(total_loss / n)
        bpp_meter.update(total_bpp / n)
        psnr_meter.update(psnr_sum / n)
        ssim_meter.update(ssim_sum / n)
        lpips_meter.update(lpips_v_sum / n)
        mse_meter.update(total_mse / n)

    metrics = {'loss': loss_meter.avg, 'bpp': bpp_meter.avg,
               'psnr': psnr_meter.avg, 'ssim': ssim_meter.avg,
               'lpips': lpips_meter.avg, 'mse': mse_meter.avg}
    metrics = reduce_dict(metrics, device)

    if is_main_process():
        val_logger.info(
            f"[VAL_QP] {global_step} 32 | "
           f"BPP: {metrics['bpp']:.6f} | PSNR: {metrics['psnr']:.6f} | SSIM: {metrics['ssim']:.6f} | LPIPS: {metrics['lpips']:.6f} | LOSS: {metrics['loss']:.6f}"
        )

    # model.train()
    raw.video_model.train()
    return metrics


# =========================================================================
#  validate_multi_qp
# =========================================================================
@torch.inference_mode()
def validate_multi_qp(model, val_loader, criterion, device, config, epoch, global_step,
                      qp_list=None):
    if qp_list is None:
        qp_list = getattr(config.train, 'val_multi_qp_list', [22, 27, 32, 37, 42])

    # model.eval()
    raw = get_raw_model(model)
    raw.video_model.eval()

    # stage = raw.get_stage(epoch)
    # pnum = raw.get_pnum(stage, config.dataset.train_dataset.seq_len)
    stage = config.stage
    strategy = raw.get_training_strategy(stage)
    idx = min(len(strategy)-1, epoch)
    _, lr, seq_len, patch_height, patch_width= strategy[idx]
    
    pnum = seq_len - 1
    val_loader.dataset.set_patch_size((patch_height, patch_width))
    val_loader.dataset.set_frame_num(seq_len)

    per_qp_meters = {
        qp: {
            'loss': AverageMeter(), 'bpp': AverageMeter(),
            'psnr': AverageMeter(), 'ssim': AverageMeter(),
            'mse': AverageMeter(), 'lpips': AverageMeter(),
        }
        for qp in qp_list
    }

    if is_main_process():
        pbar = tqdm(val_loader, desc=f'Val MultiQP step {global_step}', ncols=150)
    else:
        pbar = val_loader

    for frames in pbar:
        B, T_orig, C, H, W = frames.shape
        # frames = frames.to(device)

        start = _get_frame_offset(T_orig, pnum, mode="center")
        frames = frames[:, start:start + pnum + 1]
        _, T, _, _, _ = frames.shape

        for qp in qp_list:
            qp_index = torch.full((B,), qp, dtype=torch.long, device=device)

            raw.video_model.clear_dpb()
            # raw.video_model.set_curr_poc(0)
            
            ref_frame = frames[:, 0].to(device)
            intra_out = raw.intra_model(ref_frame, qp_index)
            raw.video_model.add_ref_frame(feature=None, frame=intra_out["x_hat"].detach())

            tl, tb, tp, ts = 0.0, 0.0, 0.0, 0.0
            tm = 0.0
            tlpips = 0.0 
            num_p = 0

            for t in range(1, T):
                cur_frame = frames[:, t].to(device)
                
                cur_qp = qp_index + raw.offsets[t % 8]
                
                out_net = raw.video_model(cur_frame, cur_qp)
                
                # color space conversion
                x_ycbcr = cur_frame 
                x_ycbcr_hat = out_net['x_hat']
                x_rgb = ycbcr2rgb(x_ycbcr)
                x_rgb_hat = ycbcr2rgb(x_ycbcr_hat)
                
                # log val 
                psnr_v, ssim_v = calculate_metrics(x_rgb_hat, x_rgb)
                lpips_v = criterion.get_perceptual_loss(x_rgb_hat, x_rgb)
                mse_loss = criterion.get_distortion(x_ycbcr_hat, x_ycbcr, x_rgb_hat, x_rgb)
                # adv_loss = criterion.get_adv_loss(x_rgb_hat, x_rgb)
                loss = mse_loss + lpips_v

                tl += loss.mean().item()
                tb += out_net["bpp"].mean().item()
                tm += mse_loss.mean().item()
                tp += psnr_v
                ts += ssim_v
                tlpips += lpips_v.mean().item()
                num_p += 1

            n = max(num_p, 1)
            per_qp_meters[qp]['loss'].update(tl / n)
            per_qp_meters[qp]['bpp'].update(tb / n)
            per_qp_meters[qp]['mse'].update(tm / n)
            per_qp_meters[qp]['psnr'].update(tp / n)
            per_qp_meters[qp]['ssim'].update(ts / n)
            per_qp_meters[qp]['lpips'].update(tlpips / n)

    # ---- reduce across GPUs ----
    results = {}
    for qp in qp_list:
        m = {k: v.avg for k, v in per_qp_meters[qp].items()}
        m = reduce_dict(m, device)
        results[qp] = m

    if is_main_process():
        for qp in qp_list:
            m = results[qp]
            val_logger.info(
                f"[VAL_QP] {global_step} {qp} | "
                f"Bpp: {m['bpp']:.6f} | PSNR: {m['psnr']:.6f} | LPIPS: {m['lpips']:.6f} | "
                f"SSIM: {m['ssim']:.6f} | MSE: {m['mse']:.6f} | Loss: {m['loss']:.6f}"
            )

        save_dir = config.save_dir
        os.makedirs(save_dir, exist_ok=True)
        json_path = os.path.join(save_dir, f'val_qp_step{global_step}.json')
        with open(json_path, 'w') as f:
            json.dump({str(k): v for k, v in results.items()}, f, indent=2)
        val_logger.info(f'Per-QP validation results saved to: {json_path}')

    # model.train()
    raw.video_model.train()
    return results


# =========================================================================
#  plot_history
# =========================================================================

def plot_history(history, save_dir):
    if not is_main_process():
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    # ---- 1. 训练曲线 (per epoch) ----
    epochs = list(range(1, len(history['train_loss']) + 1))
    if len(epochs) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        for ax, key, color, ylabel, title in [
            (axes[0], 'train_loss', 'b', 'Loss', 'Training Loss'),
            (axes[1], 'train_bpp', 'g', 'BPP', 'Training BPP'),
            (axes[2], 'train_mse', 'r', 'MSE', 'Training MSE'),
            (axes[3], 'lr', 'm', 'LR', 'Learning Rate'),
        ]:
            if key in history and len(history[key]) > 0:
                ax.plot(epochs[:len(history[key])], history[key], color=color, linewidth=1)
            ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
            ax.set_title(title); ax.grid(True, alpha=0.3)
        try:
            axes[3].set_yscale('log')
        except ValueError:
            pass
        fig.suptitle('Training Metrics over Epochs', fontsize=14, fontweight='bold')
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        train_logger.info('Saved training_curves.png')
        
    # ---- 1.5 额外训练指标（lpips, adv, disc）----
    if len(epochs) > 0 and all(k in history for k in ['train_lpips', 'train_adv', 'train_disc']):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, key, color, ylabel, title in [
            (axes[0], 'train_lpips', 'c', 'LPIPS', 'Training LPIPS'),
            (axes[1], 'train_adv', 'orange', 'Adv Loss', 'Training Adversarial Loss'),
            (axes[2], 'train_disc', 'purple', 'Disc Loss', 'Training Discriminator Loss'),
        ]:
            if key in history and len(history[key]) > 0:
                ax.plot(epochs[:len(history[key])], history[key], color=color, linewidth=1)
            ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
            ax.set_title(title); ax.grid(True, alpha=0.3)
        fig.suptitle('Additional Training Losses', fontsize=14, fontweight='bold')
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, 'training_extra_losses.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        train_logger.info('Saved training_extra_losses.png')
        
    # ---- 2. QP=32 验证曲线（增加 mse、lpips）----
    if len(history.get('val_loss', [])) > 0:
        val_x = list(range(1, len(history['val_loss']) + 1))
        # 使用 2x3 布局展示所有可用指标
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        plot_items = [
            ('val_loss', 'b', 'o', 'Loss', 'Val Loss (QP=32)'),
            ('val_bpp', 'g', 'o', 'BPP', 'Val BPP (QP=32)'),
            ('val_psnr', 'c', 'o', 'PSNR (dB)', 'Val PSNR (QP=32)'),
            ('val_ssim', 'm', 'o', 'SSIM', 'Val SSIM (QP=32)'),
            ('val_mse', 'r', 'o', 'MSE', 'Val MSE (QP=32)'),
            ('val_lpips', 'orange', 'o', 'LPIPS', 'Val LPIPS (QP=32)'),
        ]
        for ax, (key, color, marker, ylabel, title) in zip(axes, plot_items):
            vals = history.get(key, [])
            ax.plot(val_x[:len(vals)], vals, f'{color}-{marker}', markersize=3, linewidth=1)
            ax.set_xlabel('Validation Event'); ax.set_ylabel(ylabel)
            ax.set_title(title); ax.grid(True, alpha=0.3)
        # 如果指标不足6个，隐藏多余子图
        if len(history.get('val_loss', [])) == 0:
            for ax in axes[len(plot_items):]:
                ax.axis('off')
        fig.suptitle('Validation (QP=32)', fontsize=14, fontweight='bold')
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, 'validation_qp32_curves.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        train_logger.info('Saved validation_qp32_curves.png')

    # ---- 3. Per-QP 指标（增加 lpips, mse, loss）----
    val_per_qp = history.get('val_per_qp', [])
    if len(val_per_qp) > 0:
        all_qps = set()
        steps = []
        for entry in val_per_qp:
            steps.append(entry['step'])
            for qp in entry['results'].keys():
                all_qps.add(qp)
        all_qps = sorted(all_qps)
        steps = np.array(steps)
        sort_idx = np.argsort(steps)
        steps = steps[sort_idx]

        # 为每种指标准备字典
        metrics_names = ['psnr', 'bpp', 'ssim', 'lpips', 'mse', 'loss']
        qp_data = {name: {q: [] for q in all_qps} for name in metrics_names}
        for idx in sort_idx:
            entry = val_per_qp[idx]
            for qp in all_qps:
                if qp in entry['results']:
                    for name in metrics_names:
                        qp_data[name][qp].append(entry['results'][qp].get(name, np.nan))
                else:
                    for name in metrics_names:
                        qp_data[name][qp].append(np.nan)

        cmap = plt.get_cmap('viridis')
        colors = [cmap(i / max(len(all_qps) - 1, 1)) for i in range(len(all_qps))]

        # 2x3 布局
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        axes = axes.flatten()
        plot_order = [
            (axes[0], qp_data['psnr'], 'PSNR (dB)', 'Per-QP PSNR over Steps'),
            (axes[1], qp_data['bpp'], 'BPP', 'Per-QP BPP over Steps'),
            (axes[2], qp_data['ssim'], 'SSIM', 'Per-QP SSIM over Steps'),
            (axes[3], qp_data['lpips'], 'LPIPS', 'Per-QP LPIPS over Steps'),
            (axes[4], qp_data['mse'], 'MSE', 'Per-QP MSE over Steps'),
            (axes[5], qp_data['loss'], 'Loss', 'Per-QP Loss over Steps'),
        ]
        for ax, qp_dict, ylabel, title in plot_order:
            for i, qp in enumerate(all_qps):
                vals = np.array(qp_dict[qp])
                valid = ~np.isnan(vals)
                if valid.sum() > 0:
                    ax.plot(steps[valid], vals[valid], 'o-', color=colors[i],
                           markersize=3, linewidth=1, label=f'QP{qp}')
            ax.set_xlabel('Step'); ax.set_ylabel(ylabel)
            ax.set_title(title); ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, ncol=2)
        # 如果指标数不足6，隐藏未使用的子图（目前刚好6个，不需要）
        fig.suptitle('Per-QP Validation Metrics over Training Steps', fontsize=14, fontweight='bold')
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, 'per_qp_metrics_over_steps.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        train_logger.info('Saved per_qp_metrics_over_steps.png')

        # RD 曲线
        fig, ax = plt.subplots(figsize=(10, 8))
        for i, idx in enumerate(sort_idx):
            entry = val_per_qp[idx]
            qps_for_rd = sorted(entry['results'].keys())
            bpps = [entry['results'][q]['bpp'] for q in qps_for_rd]
            psnrs = [entry['results'][q]['psnr'] for q in qps_for_rd]
            color = plt.get_cmap('plasma')(i / max(len(sort_idx) - 1, 1))
            ax.plot(bpps, psnrs, 'o-', color=color, markersize=4, linewidth=1,
                   alpha=0.7, label=f'Step {entry["step"]}')
            if i == len(sort_idx) - 1:
                for qp, bx, by in zip(qps_for_rd, bpps, psnrs):
                    ax.annotate(f'QP{qp}', (bx, by), textcoords="offset points",
                               xytext=(5, 5), fontsize=7, alpha=0.8)
        ax.set_xlabel('BPP'); ax.set_ylabel('PSNR (dB)')
        ax.set_title('RD Curves over Training Steps', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, ncol=2, loc='lower right')
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, 'rd_curves.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        train_logger.info('Saved rd_curves.png')

        # 最终 RD 曲线
        last_entry = val_per_qp[sort_idx[-1]]
        final_qps = sorted(last_entry['results'].keys())
        bpps_f = [last_entry['results'][q]['bpp'] for q in final_qps]
        psnrs_f = [last_entry['results'][q]['psnr'] for q in final_qps]
        ssims_f = [last_entry['results'][q]['ssim'] for q in final_qps]

        fig, ax1 = plt.subplots(figsize=(8, 6))
        ax2 = ax1.twinx()
        l1, = ax1.plot(bpps_f, psnrs_f, 'b-o', markersize=6, linewidth=2, label='PSNR')
        l2, = ax2.plot(bpps_f, ssims_f, 'r-s', markersize=6, linewidth=2, label='SSIM')
        for qp, bx, by_p, by_s in zip(final_qps, bpps_f, psnrs_f, ssims_f):
            ax1.annotate(f'QP{qp}', (bx, by_p), textcoords="offset points",
                        xytext=(8, 6), fontsize=8, color='blue')
            ax2.annotate(f'QP{qp}', (bx, by_s), textcoords="offset points",
                        xytext=(8, -12), fontsize=8, color='red')
        ax1.set_xlabel('BPP'); ax1.set_ylabel('PSNR (dB)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax2.set_ylabel('SSIM', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax1.set_title(f'Final RD Curve (Step {last_entry["step"]})', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend([l1, l2], ['PSNR', 'SSIM'], loc='lower right')
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, 'final_rd_curve.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        train_logger.info('Saved final_rd_curve.png')


# =========================================================================
#  main
# =========================================================================
def main():
    # --------------------  parameter, log setting-------------------
    parser = argparse.ArgumentParser(description="DCVC-RT Video Model Training")
    parser.add_argument("--cfg", type=str, default="./configs/dcvcrt_video.yaml")
    parser.add_argument('--gpu_ids', nargs="+", default=[0])
    parser.add_argument('--stage', type=int, default=0, help="Training stage.")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--qp', type=int, default=None, help="QP value.")
    args = parser.parse_args()

    # ---- Distributed init ----
    rank, world_size, local_rank = setup_distributed()
    is_distributed = world_size > 1

    config = load_yaml_config(args.cfg)

    config.stage = args.stage 
    config.epochs = args.epochs 
    if args.save_dir is not None:
        config.save_dir = args.save_dir
    if args.qp is not None:
        QP = args.qp
    # ---- Device ----
    if is_distributed:
        device = torch.device(f"cuda:{local_rank}")
    else:
        gpu_id = int(args.gpu_ids[0]) if args.gpu_ids else 0
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    # ---- Seed ----
    seed = getattr(config.train, 'seed', 20260320) + rank
    set_seed(seed)

    # ---- Logger (仅主进程写文件) ----
    save_dir = getattr(config, "save_dir", "./logs")
    setup_logger(save_dir, is_main=is_main_process())
    train_logger.info('Device: ' + str(device))
    if is_distributed:
        train_logger.info(f'Distributed: rank={rank}, world_size={world_size}')

    # --------------------- Create Dataset -------------------
    train_dataset = Vimeo90kDataset(**config.dataset.train_dataset)
    val_dataset = Vimeo90kDataset(**config.dataset.eval_dataset)

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None

    train_loader = DataLoader(
        train_dataset, batch_size=config.dataset.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config.dataset.num_workers,
        pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_dataset, batch_size=config.dataset.eval_batch_size,
        shuffle=False, sampler=val_sampler,
        num_workers=config.dataset.num_workers,
        pin_memory=True)

    # -------------------- Create Model --------------------------
    model = TradVideoModel(config.model).to(device)
    # criterion = VideoCriterion().to(device)
    discriminator = HiFiCConditionalDiscriminator(**config.discriminator).to(device)
    
    # --------------------- Create Loss Function ----------------
    criterion = RDPCriterion(config.losses).to(device)

    train_logger.info(f'Total intra params:  {sum(p.numel() for p in model.intra_model.parameters()):,}')
    train_logger.info(f'Trainable inter params: {sum(p.numel() for p in model.video_model.parameters() if p.requires_grad):,}')
    train_logger.info(f'Total discriminator params: {sum(p.numel() for p in discriminator.parameters() if p.requires_grad):,}')

    # ---------------------- Create Optimizer -------------------
    inter_lr = getattr(config.optim.g_optimizer, 'lr', 1e-4)
    optimizer = configure_main_optimizer(model.video_model, inter_lr)
    disc_optimizer = configure_disc_optimizer(discriminator, getattr(config.optim.d_optimizer, 'lr', 1e-4))

    # ---------------------- DDP wrap -------------------
    if is_distributed:
        model = DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True)
        discriminator = DistributedDataParallel(
            discriminator, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=False)

    # ---- Scheduler ----
    # iters_per_epoch = len(train_loader)
    # step_interval = int(getattr(config.train, 'scheduler_step_interval', 4000))

    # warmup_steps = max((config.train.warmup_epochs * iters_per_epoch) // step_interval, 1)
    # warmup_scheduler = optim.lr_scheduler.LinearLR(
    #     optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    # reduce_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.3, patience=2, threshold=0.001,
    #     threshold_mode='abs', cooldown=5,
    #     min_lr=float(getattr(config.train, 'min_lr', 1e-6)), eps=1e-08)

    # current_scheduler = warmup_scheduler
    # switched_to_reduce_lr = False

    val_multi_qp_interval = int(getattr(config.train, 'val_multi_qp_interval', 10_000))

    # ---- History ----
    history = {
        'train_loss': [], 'train_bpp': [], 'train_mse': [], 'train_adv': [], 'train_lpips': [], 'train_disc': [],
        'val_loss': [], 'val_bpp': [], 'val_psnr': [], 'val_ssim': [], 'val_lpips': [],
        'val_mse': [], 'lr': [],
        'val_per_qp': [],
    }

    # ---- Resume ----
    start_epoch = 1
    global_step = 0
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = getattr(config.train, 'early_stop_patience', 30)
    early_stop_triggered = False

    lora_ckpt = os.path.join(save_dir, 'latest_lora_checkpoint.pth.tar')
    disc_ckpt = os.path.join(save_dir, 'latest_discriminator_model.pth.tar')
    state_ckpt = os.path.join(save_dir, 'latest_training_state.pth.tar')
    
    resume = getattr(config.train, "resume", False)
    if resume and os.path.exists(lora_ckpt):
        train_logger.info('Resuming LORA from: ' + lora_ckpt)
        ck = torch.load(lora_ckpt, map_location=device, weights_only=True)
        get_raw_model(model).video_model.load_state_dict(ck, strict=False)

        train_logger.info("Resuming Discriminator from: " + disc_ckpt)
        dck = torch.load(disc_ckpt, map_location=device, weights_only=True)
        get_raw_model(discriminator).load_state_dict(dck, strict=False)

        if os.path.exists(state_ckpt):
            train_logger.info('Resuming training state: ' + state_ckpt)
            state = torch.load(state_ckpt, map_location=device, weights_only=True)
            optimizer.load_state_dict(state['optimizer_state_dict'])
            disc_optimizer.load_state_dict(state['disc_optimizer_state_dict'])
            start_epoch = state['epoch'] + 1
            global_step = state.get('global_step', 0)
            best_val_loss = state.get('best_val_loss', float('inf'))
            history = state.get('history', history)
            
            if 'val_per_qp' not in history:
                history['val_per_qp'] = []
                
            patience_counter = state.get('patience_counter', 0)
            # ------------------------ scheduler -------------------
            # if start_epoch > config.train.warmup_epochs:
            #     current_scheduler = reduce_scheduler
            #     switched_to_reduce_lr = True
            # try:
            #     current_scheduler.load_state_dict(state.get('warmup_scheduler_state_dict', {}))
            #     reduce_scheduler.load_state_dict(state.get('reduce_scheduler_state_dict', {}))
            # except Exception:
            #     train_logger.info('Warning: could not restore scheduler state.')
            
            train_logger.info(f'Resumed: epoch={start_epoch}, step={global_step}, '
                              f'best_val_loss={best_val_loss:.4f}, '
                              f"patience={patience_counter}")

    # =====================================================================
    #  Main Training Loop
    # =====================================================================
    train_logger.info('========== Start Training ==========')
    epochs = args.epochs
    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()

        if is_distributed:
            train_sampler.set_epoch(epoch)

        train_metrics, global_step, val_metrics_list, val_multi_qp_list = \
            train_one_epoch(
                model, discriminator, criterion, optimizer, disc_optimizer, train_loader, val_loader,
                device, config, epoch,
                scheduler=None,
                step_interval=None,
                global_step=global_step,
                val_multi_qp_interval=val_multi_qp_interval,
            )
        train_loss, train_bpp, train_mse = train_metrics['loss'], train_metrics['bpp'], train_metrics['mse']
        train_adv, train_lpips, train_disc = train_metrics['adv'], train_metrics['lpips'], train_metrics['disc']
        
        # ---- process validation results ----
        raw = get_raw_model(model)
        for vm in val_metrics_list:
            if vm['loss'] < best_val_loss:
                best_val_loss = vm['loss']
                patience_counter = 0
                train_logger.info(f'  *** New best val loss: {best_val_loss:.6f} ***')
                save_checkpoint(raw.video_model, discriminator, optimizer, disc_optimizer, None, None,
                                epoch, global_step, best_val_loss, history, config,
                                is_best=True)
            else:
                patience_counter += 1
                train_logger.info(f'  No improvement for {patience_counter} val cycles')

            if patience_counter >= early_stop_patience:
                train_logger.info(f'  *** Early stopping at epoch {epoch} ***')
                early_stop_triggered = True

        # ---- history ----
        history['train_loss'].append(train_loss)
        history['train_bpp'].append(train_bpp)
        history['train_mse'].append(train_mse)
        history['train_adv'].append(train_adv)
        history['train_lpips'].append(train_lpips)
        history['train_disc'].append(train_disc)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        if val_multi_qp_list:
            history['val_per_qp'].extend(val_multi_qp_list)

        if val_metrics_list:
            last_val = val_metrics_list[-1]
            history['val_loss'].append(last_val['loss'])
            history['val_bpp'].append(last_val['bpp'])
            history['val_psnr'].append(last_val['psnr'])
            history['val_ssim'].append(last_val['ssim'])
            history['val_mse'].append(last_val.get('mse', 0.0))
            history['val_lpips'].append(last_val.get('lpips', 0.0))
        else:
            history['val_loss'].append(history['val_loss'][-1] if history['val_loss'] else 0.0)
            history['val_bpp'].append(history['val_bpp'][-1] if history['val_bpp'] else 0.0)
            history['val_psnr'].append(history['val_psnr'][-1] if history['val_psnr'] else 0.0)
            history['val_ssim'].append(history['val_ssim'][-1] if history['val_ssim'] else 0.0)
            history['val_lpips'].append(history['val_lpips'][-1] if history['val_lpips'] else 0.0)
            history['val_mse'].append(0.0)

        # ---- epoch summary ----
        et = time.time() - epoch_start
        train_logger.info(f'Epoch {epoch:03d}/{config.train.epochs} | Time: {et:.1f}s')
        train_logger.info(f'  Train Loss: {train_loss:.4f} | BPP: {train_bpp:.4f} | MSE: {train_mse:.6f}')
        if val_metrics_list:
            lv = val_metrics_list[-1]
            train_logger.info(f'  Val   Loss: {lv["loss"]:.4f} | BPP: {lv["bpp"]:.4f} | '
                              f'PSNR: {lv["psnr"]:.2f} dB | SSIM: {lv["ssim"]:.4f}')
        train_logger.info(f'  LR: {optimizer.param_groups[0]["lr"]:.6f} | '
                          f'Stage: {get_raw_model(model).current_stage}')

        # ---- save ----
        save_checkpoint(raw.video_model, discriminator, optimizer, disc_optimizer, None, None,
                        epoch, global_step, best_val_loss, history, config, is_best=False)

        # ---------------- scheduler -------------------
        # if not switched_to_reduce_lr and epoch >= config.train.warmup_epochs:
        #     train_logger.info('  *** Switching to ReduceLROnPlateau ***')
        #     current_scheduler = reduce_scheduler
        #     switched_to_reduce_lr = True

        # if early_stop_triggered:
        #     break

        torch.cuda.empty_cache()

    cleanup_distributed()
    # =====================================================================
    #  Done
    # =====================================================================
    train_logger.info('========== Training Complete ==========')
    plot_history(history, save_dir)
    


if __name__ == "__main__":
    main()
