import os
import sys
import random
import json
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# from src.models.image_model import DCVCRTImage
# from src.models.lora_image_model import DCVCRTImage
from src.models.moe_lora_image_model import DCVCRTImage
# from src.discriminator.hific_discriminator import HiFiCDiscriminator, HiFiCConditionalDiscriminator
from src.discriminator.module_list_discriminator import ModuleListDiscriminator
import src.loralib as loralib 
# from src.losses.rd_loss import MultiQPRateDistortionLoss
from src.dataset.vimeo90k_dataset import Vimeo90kDataset
from src.dataset.imagenet_dataset import ImageNetDataset
from src.utils.transforms import rgb2ycbcr, ycbcr2rgb, yuv_444_to_420
from src.metrics.metric import calculate_metrics
import yaml
from configs.config import load_yaml_config, DotDict
from src.utils.utils import set_seed, AverageMeter, get_state_dict
import argparse
from src.losses.perceptual_loss import LPIPSLoss
from src.losses.gan_loss import VanillaGANLoss


# class Criterion(nn.Module):
#     def __init__(self, lambdas=None, metrics="mse"):
#         super().__init__()
#         self.metrics = metrics
#         self.mse = nn.MSELoss(reduction="none")
#         # 将配置文件中的 lambda 列表转为 tensor，固定在显存中
#         # [lambda0, lambda1] = [1, 768], lambda = e^(ln(lambda0) + qt/(q_num) * (ln(lambda1) - ln(lambda0)))
#         if lambdas is None:
#             # l_min, l_max = 1, 768
#             l_min, l_max = 45, 4200
#             qt = np.linspace(0, 63, 64)
#             lambdas = np.exp(np.log(l_min) + qt / (64 - 1) * (np.log(l_max) - np.log(l_min)))
#         self.register_buffer("lambdas", torch.tensor(lambdas, dtype=torch.float32))

#     def forward(self, output, target, qp_index, weight=1.0, color_space="ycbcr"):
#         res = {}
#         x_hat = output["x_hat"]
#         res["bpp"] = output["bpp"]
        
#         # if color_space.lower() == "ycbcr":
#         # y_hat, uv_hat = yuv_444_to_420(x_hat)
#         # y_tgt, uv_tgt = yuv_444_to_420(target)
        
#         # mse_y = self.mse(y_hat, y_tgt).mean(dim=[1, 2, 3])
#         # mse_u = self.mse(uv_hat[:, 0:1, :, :], uv_tgt[:, 0:1, :, :]).mean(dim=[1, 2, 3])
#         # mse_v = self.mse(uv_hat[:, 1:2, :, :], uv_tgt[:, 1:2, :, :]).mean(dim=[1, 2, 3])
#         # yuv420_mse = (6 * mse_y + mse_u + mse_v) / 8.0
        
#         yuv420_mse = self.mse(x_hat, target).mean(dim=[1, 2, 3])
    
#         rgb_hat = ycbcr2rgb(x_hat)
#         rgb_tgt = ycbcr2rgb(target)
#         rgb_mse = self.mse(rgb_hat, rgb_tgt).mean(dim=[1, 2, 3])
#         k = 0.8
#         res["mse_loss"] = k * yuv420_mse + (1 - k) * rgb_mse
        
#         # else:
#         #     res["mse_loss"] = self.mse(x_hat, target).mean(dim=[1, 2, 3])
        
#         res["loss"] = self.lambdas[qp_index.long()] * res["mse_loss"] + res["bpp"]
        
#         return {
#             "bpp": res["bpp"],
#             "mse": res["mse_loss"],
#             "loss": res["loss"],
#         }
        

class RDPCriterion(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt 
        self.mse = nn.MSELoss(reduction='none')
        self.gan_loss = VanillaGANLoss(**opt.gan_loss)
        self.perceptual_loss = LPIPSLoss(**opt.perceptual_loss)
        self.percept_weight = opt.perceptual_loss_weight 
        self.gen_weight = opt.gen_loss_weight
        self.distortion_weight = opt.distortion_loss_weight
        lmbs = get_training_lambdas([1, 7.5], 64)
        self.register_buffer("lambdas", torch.tensor(lmbs))
        
        
    def compression_loss(self, x_ycbcr_hat, x_ycbcr, x_rgb_hat, x_rgb, d_fake, bpp, qp_index):
        # Optimize Generator 
        res = {}
        
        yuv420_mse = self.mse(x_ycbcr_hat, x_ycbcr).mean(dim=[1, 2, 3])
        # rgb_hat = ycbcr2rgb(x_hat)
        # rgb_tgt = ycbcr2rgb(target)
        rgb_mse = self.mse(x_rgb_hat, x_rgb).mean(dim=[1, 2, 3])
        k = 0.8
        res["mse_loss"] = (k * yuv420_mse + (1 - k) * rgb_mse) * self.distortion_weight
        
        batch_size = x_rgb.shape[0]
        # LPIPS returns [B, 1, 1, 1]; reduce only non-batch dimensions.
        res["perceptual_loss"] = self.perceptual_loss(x_rgb_hat, x_rgb).reshape(batch_size, -1).mean(dim=1) * self.percept_weight
        
        res["adv_loss"] = self.gan_loss(d_fake, is_real=True, is_disc=False).reshape(batch_size, -1).mean(dim=1) * self.gen_weight
        
        # res["loss"] = (res["mse_loss"] + res["perceptual_loss"] + res["adv_loss"]) * self.lambdas[qp_index.long()]   # + bpp 
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
    
    
def get_training_lambdas(lambdas, qp_num):
    all_lambdas = np.linspace(np.log(lambdas[0]), np.log(lambdas[1]), qp_num)
    all_lambdas = np.exp(all_lambdas)
    return all_lambdas
  

def setup_distributed():
    """初始化分布式训练环境"""
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


def setup_logger(save_dir, is_main):
    """分别设置训练日志和验证日志"""
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
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        train_logger.addHandler(ch)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            train_log_path = os.path.join(save_dir, 'train.log')
            fh = logging.FileHandler(train_log_path)
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            train_logger.addHandler(fh)

            val_log_path = os.path.join(save_dir, 'val.log')
            vh = logging.FileHandler(val_log_path)
            vh.setLevel(logging.INFO)
            vh.setFormatter(formatter)
            val_logger.addHandler(vh)

    return train_logger, val_logger


train_logger = logging.getLogger("train_logger")
val_logger = logging.getLogger("val_logger")


def get_raw_model(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def reduce_dict(metrics, device):
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return metrics
    world_size = dist.get_world_size()
    reduced = {}
    for k, v in metrics.items():
        tensor = torch.tensor(v, device=device, dtype=torch.float32)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        reduced[k] = tensor.item() / world_size
    return reduced



def enable_optimizer_parameters(model, optimizer):
    """Restore only optimizer-owned parameters after the discriminator phase."""
    model.requires_grad_(False)
    # Clear stale gradients outside the optimizer too (e.g. after an older run).
    model.zero_grad(set_to_none=True)
    parameters = list(dict.fromkeys(
        p for group in optimizer.param_groups for p in group["params"]
    ))
    for parameter in parameters:
        parameter.requires_grad_(True)
    return parameters


@torch.no_grad()
def discriminator_real_reference(model, images, qp_index, original_rgb):
    """Use QP+8 reconstructions as positives where that QP is available."""
    reference = original_rgb.detach().clone()
    valid = qp_index + 8 <= 63
    if not valid.any():
        return reference
    was_training = model.training
    try:
        # Deterministic routing; reference construction must not train the codec.
        model.eval()
        out = model(images[valid], qp_index[valid] + 8)
        reference[valid] = ycbcr2rgb(out["x_hat"])
    finally:
        model.train(was_training)
    return reference


def train_one_epoch(model, disc_model, criterion, optimizer, disc_optimizer, train_loader, val_loader, config, epoch,
                    scheduler=None, step_interval=1, global_step=0, val_multi_qp_interval=None):

    model.train()
    disc_model.train()
    
    device = next(model.parameters()).device
    
    epoch_loss_meter = AverageMeter()
    epoch_bpp_meter = AverageMeter()
    epoch_mse_meter = AverageMeter()
    epoch_disc_meter = AverageMeter()
    epoch_perc_meter = AverageMeter()
    epoch_adv_meter = AverageMeter()
    

    per_qp_meters = {
        qp: {'loss': AverageMeter(), 'mse': AverageMeter(), 'bpp': AverageMeter()}
        for qp in range(64)
    }

    val_metrics_list = []
    val_multi_qp_list = []

    if is_main_process():
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}', ncols=150)
    else:
        pbar = train_loader

    for batch_idx, d in enumerate(pbar):
        d = d.to(device)
        
        # =========================================================
        # step 1：Train Generator G
        # =========================================================   
        disc_model.requires_grad_(False)     
        generator_parameters = enable_optimizer_parameters(model, optimizer)
        optimizer.zero_grad(set_to_none=True)

        
        # qp_index = torch.randint(0, 64, (d.size(0),), device=device)
        qp_val = np.random.randint(0, 64)
        qp_index = torch.full((d.size(0),), qp_val, dtype=torch.long, device=device)
        
        out_net = model(d, qp_index)
        
        # ------------ convert color space ---------------------
        x_ycbcr = d 
        x_ycbcr_hat = out_net["x_hat"]
        x_rgb, x_rgb_hat = ycbcr2rgb(x_ycbcr), ycbcr2rgb(x_ycbcr_hat)
        
        d_fake = disc_model(x_rgb_hat, qp_val, y_hat=out_net["y_hat"])
        
        out_criterion = criterion.compression_loss(x_ycbcr_hat, x_ycbcr, x_rgb_hat, x_rgb, d_fake, out_net["bpp"], qp_index)

        loss = out_criterion["loss"].mean() + out_net["moe_loss"] * 0.1
        loss.backward()

        if getattr(config.train, 'clip_max_norm', None) is not None:
            torch.nn.utils.clip_grad_norm_(generator_parameters, config.train.clip_max_norm)

        optimizer.step()
        lr = optimizer.param_groups[0]["lr"]

        # log loss item
        mse_val = out_criterion["mse_loss"].mean().item()
        perc_val = out_criterion["perceptual_loss"].mean().item()
        adv_val = out_criterion["adv_loss"].mean().item()
        epoch_perc_meter.update(perc_val)
        epoch_adv_meter.update(adv_val)

        # =========================================================
        # step 2：Train Discriminator D
        # =========================================================       
        disc_model.requires_grad_(True)
        model.requires_grad_(False)
        disc_optimizer.zero_grad(set_to_none=True)
        
        # Keep x_rgb as the original target for generator MSE/LPIPS.
        # Both discriminator branches retain the current-QP head and latent.
        x_rgb_real = discriminator_real_reference(model, d, qp_index, x_rgb)
        d_real = disc_model(x_rgb_real, qp_val, y_hat=out_net["y_hat"].detach())
        d_fake = disc_model(x_rgb_hat.detach(), qp_val, y_hat=out_net["y_hat"].detach())
        
        
        d_out_criterion = criterion.disc_loss(d_real, d_fake)
        
        d_loss = d_out_criterion["disc_loss"].mean()
        d_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(disc_model.parameters(), max_norm=1.0)
        
        disc_optimizer.step()
         
        global_step += 1

        # log loss item 
        epoch_loss_meter.update(loss.item())
        # epoch_bpp_meter.update(out_criterion["bpp"].mean().item())
        epoch_bpp_meter.update(out_net["bpp"].mean().item())
        epoch_mse_meter.update(mse_val)
        epoch_disc_meter.update(d_loss.item())
        

        # log by qp
        for qp in torch.unique(qp_index, sorted=True):
            qp_val = qp.item()
            mask = (qp_index == qp)
            n_qp = mask.sum().item()
            per_qp_meters[qp_val]['loss'].update(out_criterion["loss"][mask].mean().item(), n_qp)
            per_qp_meters[qp_val]['mse'].update(out_criterion["mse_loss"][mask].mean().item(), n_qp)
            per_qp_meters[qp_val]['bpp'].update(out_net["bpp"][mask].mean().item(), n_qp)

        # scheduler step：每 step_interval 次 iteration 执行一次
        if scheduler is not None and not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            if global_step % step_interval == 0:
                scheduler.step()

        # 每隔 100 iter 记录一次 per-qp 训练指标到 train.log
        if global_step % 10000 == 0 and is_main_process():
            for qp in range(64):
                if per_qp_meters[qp]['loss'].count > 0:
                    train_logger.info(
                        f"[TRAIN_QP] {global_step} {qp} "
                        f"{per_qp_meters[qp]['loss'].avg:.6f} "
                        f"{per_qp_meters[qp]['mse'].avg:.6f} "
                        f"{per_qp_meters[qp]['bpp'].avg:.6f}"
                    )
            # 记录完成后重置 per-qp 统计器
            per_qp_meters = {
                qp: {'loss': AverageMeter(), 'mse': AverageMeter(), 'bpp': AverageMeter()}
                for qp in range(64)
            }

        # 每隔 100000 iter 进行一次 Single QP 验证（跳过多 QP 验证重合的步数，避免重复）
        skip_cur_qp = (val_multi_qp_interval is not None and global_step % val_multi_qp_interval == 0)
        if global_step % 5_000 == 0 and not skip_cur_qp:
            val_metrics = validate_single_qp(model, val_loader, criterion, device, config, global_step)
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])
            val_metrics_list.append(val_metrics)
            # save_model(model, config, "middle_iter_model.pth.tar")            # NOTE: New add
            save_lora_checkpoint(model, config, bias="lora_only", filename="middle_iter_lora_model.pth.tar")
            

        # 每隔 val_multi_qp_interval 步进行一次多 QP 验证
        if val_multi_qp_interval is not None and global_step % val_multi_qp_interval == 0:
            multi_qp_results = validate_multi_qp(
                model, val_loader, criterion, device, config, global_step
            )
            val_multi_qp_list.append({'step': global_step, 'results': multi_qp_results})

        if is_main_process():
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'bpp': f'{out_net["bpp"].mean().item():.4f}',
                'mse': f'{mse_val:.6f}',
                'percept': f'{perc_val:.4f}',
                'adv': f'{adv_val:.4f}',
                'd_loss': f'{d_loss.item():.4f}',
                'lr': f'{lr:.6f}'
            })
        
        # if not switched_to_reduce_lr and global_step > warmup_steps * step_interval:
        #     train_logger.info('  *** Switching to ReduceLROnPlateau scheduler ***')
        #     current_scheduler = reduce_lr_scheduler
        #     switched_to_reduce_lr = True

    metrics = {
        'loss': epoch_loss_meter.avg,
        'bpp': epoch_bpp_meter.avg,
        'mse': epoch_mse_meter.avg,
        'd_loss': epoch_disc_meter.avg,
        'perc': epoch_perc_meter.avg,
        'adv': epoch_adv_meter.avg
    }
    metrics = reduce_dict(metrics, device)

    return metrics, global_step, val_metrics_list, val_multi_qp_list


@torch.inference_mode()
def validate_single_qp(model, val_loader, criterion, device, config, global_step):
    """只验证 single qp，记录 bpp/psnr/ssim/loss 到 val.log"""
    model.eval()
    loss_meter = AverageMeter()
    bpp_meter = AverageMeter()
    mse_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    lpips_meter = AverageMeter()
    
    percept_weight = criterion.percept_weight
    qp_val = 32

    if is_main_process():
        pbar = tqdm(val_loader, desc=f'Val QP {qp_val} step {global_step}')
    else:
        pbar = val_loader

    for d in pbar:
        d = d.to(device)
        # qp_index = torch.full((d.size(0),), 32, dtype=torch.long, device=device)
        qp_index = torch.full((d.size(0),), qp_val, dtype=torch.long, device=device)
        out_net = model(d, qp_index)

        # convert color space 
        x_ycbcr = d
        x_ycbcr_hat = out_net["x_hat"]
        x_rgb, x_rgb_hat = ycbcr2rgb(x_ycbcr), ycbcr2rgb(x_ycbcr_hat)
        
        yuv420_mse = criterion.mse(x_ycbcr_hat, x_ycbcr).mean(dim=[1,2,3])
        rgb_mse    = criterion.mse(x_rgb_hat, x_rgb).mean(dim=[1,2,3])
        mse_loss   = (0.8 * yuv420_mse + 0.2 * rgb_mse).mean().item() * criterion.distortion_weight
        lpips_val = criterion.perceptual_loss(x_rgb_hat, x_rgb)
        lpips_loss = lpips_val.mean().item()

        total_loss = mse_loss + percept_weight * lpips_loss
        
        psnr, ssim_val = calculate_metrics(x_rgb_hat, x_rgb)

        bpp = out_net["bpp"].mean().item()
        
        loss_meter.update(total_loss)
        bpp_meter.update(bpp)
        mse_meter.update(mse_loss)
        psnr_meter.update(psnr)
        ssim_meter.update(ssim_val)
        lpips_meter.update(lpips_loss * percept_weight)

    metrics = {
        'loss': loss_meter.avg,
        'bpp': bpp_meter.avg,
        'mse': mse_meter.avg,
        'psnr': psnr_meter.avg,
        'ssim': ssim_meter.avg,
        'lpips': lpips_meter.avg
    }
    metrics = reduce_dict(metrics, device)

    if is_main_process():
        val_logger.info(
            f"[VAL_QP] {global_step} {qp_val} "
            f"{metrics['bpp']:.6f} {metrics['psnr']:.6f} "
            f"{metrics['ssim']:.6f} {metrics['lpips']:.6f} {metrics['loss']:.6f}"
        )

    model.train()
    return metrics


@torch.inference_mode()
def validate_multi_qp(model, val_loader, criterion, device, config, global_step, qp_list=None):
    """验证多个 QP，记录每个 QP 的 bpp/psnr/ssim/mse/loss 到 val.log"""
    if qp_list is None:
        qp_list = getattr(config.train, 'val_multi_qp_list', [22, 27, 32, 37, 42])

    model.eval()
    percept_weight = criterion.percept_weight

    per_qp_meters = {
        qp: {
            'loss': AverageMeter(), 'bpp': AverageMeter(),
            'psnr': AverageMeter(), 'ssim': AverageMeter(),
            'mse': AverageMeter(), 'lpips': AverageMeter()
        }
        for qp in qp_list
    }

    if is_main_process():
        pbar = tqdm(val_loader, desc=f'Val MultiQP step {global_step}', ncols=150)
    else:
        pbar = val_loader

    for d in pbar:
        d = d.to(device)
        for qp in qp_list:
            qp_index = torch.full((d.size(0),), qp, dtype=torch.long, device=device)
            out_net = model(d, qp_index)
            
            # convert color space 
            x_ycbcr = d
            x_ycbcr_hat = out_net["x_hat"]
            x_rgb, x_rgb_hat = ycbcr2rgb(x_ycbcr), ycbcr2rgb(x_ycbcr_hat)
            
            yuv420_mse = criterion.mse(x_ycbcr_hat, x_ycbcr).mean(dim=[1,2,3])
            rgb_mse    = criterion.mse(x_rgb_hat, x_rgb).mean(dim=[1,2,3])
            mse_loss   = (0.8 * yuv420_mse + 0.2 * rgb_mse).mean().item() * criterion.distortion_weight

            lpips_val = criterion.perceptual_loss(x_rgb_hat, x_rgb)
            lpips_loss = lpips_val.mean().item()

            total_loss = mse_loss + percept_weight * lpips_loss

            psnr, ssim_val = calculate_metrics(x_rgb_hat, x_rgb)
            bpp = out_net["bpp"].mean().item()

            # log items 
            per_qp_meters[qp]['loss'].update(total_loss)
            per_qp_meters[qp]['bpp'].update(bpp)
            per_qp_meters[qp]['mse'].update(mse_loss)
            per_qp_meters[qp]['psnr'].update(psnr)
            per_qp_meters[qp]['ssim'].update(ssim_val)
            per_qp_meters[qp]['lpips'].update(lpips_loss * percept_weight)
            
    # Reduce across distributed processes
    results = {}
    for qp in qp_list:
        m = {k: v.avg for k, v in per_qp_meters[qp].items()}
        m = reduce_dict(m, device)
        results[qp] = m

    if is_main_process():
        for qp in qp_list:
            m = results[qp]
            val_logger.info(
                f"[VAL_QP] {global_step} {qp} "
                f"{m['bpp']:.6f} {m['psnr']:.6f} "
                f"{m['ssim']:.6f} {m['lpips']:.6f} {m['loss']:.6f}"
            )

        # 保存 per-QP 结果为 JSON 方便离线分析 RD 曲线
        save_dir = config.save_dir
        os.makedirs(save_dir, exist_ok=True)
        json_path = os.path.join(save_dir, f'val_qp_step{global_step}.json')
        # Convert int keys to str for JSON serialization
        json_results = {str(qp): v for qp, v in results.items()}
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        val_logger.info(f'Per-QP validation results saved to: {json_path}')

    model.train()
    return results


def save_model(model, config, filename='latest_model.pth.tar'):
    if not is_main_process():
        return
    save_dir = config.save_dir
    os.makedirs(save_dir, exist_ok=True)
    raw_model = get_raw_model(model)
    model_path = os.path.join(save_dir, filename)
    if hasattr(raw_model, "update"):
        raw_model.update()          # ignore 
    torch.save(raw_model.state_dict(), model_path)
    train_logger.info('Model saved to: ' + model_path)


def save_lora_checkpoint(model, config, bias="lora_only", filename='latest_lora_checkpoint.pth.tar'):
    if not is_main_process():
        return 
    save_dir = config.save_dir 
    os.makedirs(save_dir, exist_ok=True)
    lora_checkpoint = loralib.lora_state_dict(model, bias=bias)
    torch.save(lora_checkpoint, os.path.join(save_dir, filename))
    train_logger.info('Lora checkpoint saved to: ' + os.path.join(save_dir, filename))
    

def save_training_state(optimizer, disc_optimizer, scheduler, epoch, global_step, best_val_loss, history, config,
                        filename='latest_training_state.pth.tar'):
    if not is_main_process():
        return
    save_dir = config.save_dir
    os.makedirs(save_dir, exist_ok=True)
    state = {
        'epoch': epoch,
        'global_step': global_step,
        'best_val_loss': best_val_loss,
        'optimizer_state_dict': optimizer.state_dict(),
        'disc_optimizer_state_dict': disc_optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'history': history,
    }
    state_path = os.path.join(save_dir, filename)
    torch.save(state, state_path)
    train_logger.info('Training state saved to: ' + state_path)


def save_checkpoint(model, disc_model, optimizer, disc_optimizer, scheduler, epoch, global_step, best_val_loss, history, config, bias="lora_only", is_best=False):
    """保存最新模型和训练状态，模型和训练状态分开存储"""
    # save_model(model, config, 'latest_model.pth.tar')
    save_lora_checkpoint(model, config, bias=bias, filename="latest_lora_checkpoint.pth.tar")
    save_model(disc_model, config, "latest_disc_model.pth.tar")
    save_training_state(optimizer, disc_optimizer, scheduler, epoch, global_step, best_val_loss, history, config,
                          'latest_training_state.pth.tar')

    if is_best:
        # save_model(model, config, 'best_model.pth.tar')
        save_lora_checkpoint(model, config, bias=bias, filename="best_lora_checkpoint.pth.tar")
        train_logger.info('Best lora checkpoint saved to: ' + os.path.join(config.save_dir, 'best_lora_checkpoint.pth.tar'))


def configure_main_optimizer(model, inter_lr):
    # amortization_parameters = [p for n, p in model.named_parameters() if "bit_estimator_z" not in n and "gaussian_encoder" not in n and p.requires_grad]
    # hyper_parameters = [p for n, p in model.named_parameters() if ("bit_estimator_z" in n or "gaussian_encoder" in n) and p.requires_grad]

    # inter_params = set(amortization_parameters) & set(hyper_parameters)
    # union_params = set(amortization_parameters) | set(hyper_parameters)

    # assert len(inter_params) == 0 and len(union_params) == len(list(p for p in model.parameters() if p.requires_grad))

    # params_group = [
    #     {"params": amortization_parameters, "lr": inter_lr},
    #     {"params": hyper_parameters, "lr": inter_lr},
    # ]
    optimizer = torch.optim.AdamW([param for param in model.parameters() if param.requires_grad], lr=inter_lr)
    return optimizer


def configure_disc_optimizer(model, lr):
    disc_optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return disc_optimizer 



def main():
    args = argparse.ArgumentParser()
    args.add_argument('--cfg', type=str, default="./configs/dcvcrt_image.yaml", help="config yaml file path.")
    args.add_argument('--save_dir', type=str, default="", help="save directory.")
    args = args.parse_args()

    rank, world_size, local_rank = setup_distributed()
    is_distributed = world_size > 1

    config_path = args.cfg
    config = load_yaml_config(config_path)

    set_seed(config.train.seed + rank)
    
    if args.save_dir != "":
        config.save_dir = args.save_dir

    if is_distributed:
        device = torch.device(f"cuda:{local_rank}")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.train.gpu_id)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_main_process():
        os.makedirs(config.save_dir, exist_ok=True)
    setup_logger(config.save_dir, is_main_process())

    if is_main_process():
        train_logger.info('Using device: ' + str(device))
        if torch.cuda.is_available():
            train_logger.info('GPU: ' + torch.cuda.get_device_name(local_rank if is_distributed else 0))

        config_save_path = os.path.join(config.save_dir, 'config.json')
        with open(config_save_path, 'w') as f:
            json.dump(dict(config), f, indent=4, default=str)
        train_logger.info('Config saved to: ' + config_save_path)
        train_logger.info('Loading datasets...')

    # ------------------- Create Dataset & DataLoader --------------- 
    # train_dataset = Vimeo90kDataset(**config.dataset.train_dataset)
    # test_dataset = Vimeo90kDataset(**config.dataset.eval_dataset)
    train_dataset = ImageNetDataset(**config.dataset.train_dataset)
    test_dataset = ImageNetDataset(**config.dataset.eval_dataset)

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed else None
    val_sampler = DistributedSampler(test_dataset, shuffle=False) if is_distributed else None

    train_loader = DataLoader(train_dataset, batch_size=config.dataset.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=config.dataset.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(test_dataset, batch_size=config.dataset.eval_batch_size, shuffle=False, sampler=val_sampler, num_workers=config.dataset.num_workers, pin_memory=True)

    if is_main_process():
        train_logger.info('Initializing model...')

    # -----------------  Create Model ------------------ 
    bias = "lora_only"
    model = DCVCRTImage(N=config.model.N, z_channel=config.model.z_channel).to(device)
    loralib.mark_only_lora_as_trainable(model, bias=bias)
    model.bit_estimator_z.requires_grad_(False)             # bit_estimator_z is not trainable
    if hasattr(model, "update"):
        model.update() # not return 
    
    # save: lora_state = loralib.lora_state_dict(model, bias=bias); torch.save(lora_state, save_path)
    
    # disc_model = HiFiCConditionalDiscriminator(**config.discriminator).to(device)
    disc_model = ModuleListDiscriminator(**config.discriminator).to(device)
    
    # ------------------ Load Pretrained Checkpoint ------------------ 
    if is_main_process() and getattr(config.model, "pretrained", None) is not None:
        ckpt = get_state_dict(config.model.pretrained)
        model.load_state_dict(ckpt, strict=False)
        print("Load pretrained image model from: ", config.model.pretrained)

    if is_main_process():
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        train_logger.info('Total parameters: %.2f M' % (total_params / 1e6))
        train_logger.info('Trainable parameters: %.2f M' % (trainable_params / 1e6))

    # ------------------- Create Loss Function ------------------
    # criterion = MultiQPRateDistortionLoss().to(device)
    # criterion = Criterion().to(device)
    criterion = RDPCriterion(config.losses).to(device)

    # -------------------- Create Optimizer ----------------------
    # lr = config.optim.g_optimizer.lr
    optimizer = configure_main_optimizer(model, config.optim.g_optimizer.lr)
    disc_optimizer = configure_disc_optimizer(disc_model, config.optim.d_optimizer.lr)

    # DDP wrap 
    if is_distributed:
        model = DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True
        )
        disc_model = DistributedDataParallel(
            disc_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
        )

    # scheduler step 间隔（按 iteration）
    step_interval = int(getattr(config.train, 'scheduler_step_interval', 100))
    iters_per_epoch = len(train_loader)

    # Warmup phase using LinearLR（按 iteration step，total_iters 根据 step_interval 折算）
    warmup_steps = max((config.train.warmup_epochs * iters_per_epoch) // step_interval, 1)
    # warmup_steps = config.train.get('warmup_steps', 1000)
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps
    )
    # Main scheduler using ReduceLROnPlateau（mode='min'，监控 val_loss）
    reduce_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.3,
        patience=3,
        threshold=0.001,
        threshold_mode='abs',
        cooldown=5,
        min_lr=float(config.train.get('min_lr', 1e-7)),
        eps=1e-08
    )
    current_scheduler = warmup_scheduler
    switched_to_reduce_lr = False

    history = {
        'train_loss': [],
        'train_bpp': [],
        'train_mse': [],
        'train_d_loss': [],
        'train_perc': [],
        'train_adv': [],
        'val_loss': [],
        'val_bpp': [],
        'val_psnr': [],
        'val_ssim': [],
        'val_mse': [],
        'val_lpips': [],
        'lr': [],
        'val_per_qp': [],  # list of {step, results: {qp: {bpp, psnr, ssim, mse, loss}}}
    }

    start_epoch = 1
    global_step = 0
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 30
    early_stop_triggered = False

    # 多 QP 验证间隔 and scheduler switch interval 
    val_multi_qp_interval = int(getattr(config.train, 'val_multi_qp_interval', 100_000))

    # ------------------------- Resume ----------------------------------
    # Resume：模型和训练状态分别加载
    model_ckpt_path = os.path.join(config.save_dir, 'latest_lora_checkpoint.pth.tar')
    disc_model_ckpt_path = os.path.join(config.save_dir, 'latest_disc_model.pth.tar')
    state_ckpt_path = os.path.join(config.save_dir, 'latest_training_state.pth.tar')

    if os.path.exists(model_ckpt_path) and config.train.resume:
        if is_main_process():
            train_logger.info('Found lora model checkpoint, resuming: ' + model_ckpt_path)
        checkpoint = torch.load(model_ckpt_path, map_location=device, weights_only=True)
        get_raw_model(model).load_state_dict(checkpoint, strict=False)                  # lora weights
        # ckpt = get_state_dict(config.model.pretrained)
        # get_raw_model(model).load_state_dict(ckpt, strict=False)     # original weights

        disc_checkpoint = torch.load(disc_model_ckpt_path, map_location=device, weights_only=True)
        get_raw_model(disc_model).load_state_dict(disc_checkpoint)

        if os.path.exists(state_ckpt_path):
            if is_main_process():
                train_logger.info('Found training state, resuming: ' + state_ckpt_path)
            state = torch.load(state_ckpt_path, map_location=device, weights_only=True)
            optimizer.load_state_dict(state['optimizer_state_dict'])
            if 'disc_optimizer_state_dict' in state:
                disc_optimizer.load_state_dict(state['disc_optimizer_state_dict'])
            else:
                if is_main_process():
                    train_logger.info("Warning: disc_optimizer state not found in checkpoint, using fresh state.")
            start_epoch = state['epoch'] + 1
            global_step = state.get('global_step', 0)
            best_val_loss = state.get('best_val_loss', float('inf'))
            
            # history = state.get('history', history)
            # if 'val_per_qp' not in history:
            #     history['val_per_qp'] = []
            
            history = state.get('history', history)
            for key in ['train_perc', 'train_adv', 'val_lpips', 'val_per_qp']:
                if key not in history:
                    history[key] = []

            if start_epoch > config.train.warmup_epochs:
            # if global_step > warmup_steps * step_interval:
                current_scheduler = reduce_lr_scheduler
                switched_to_reduce_lr = True

            try:
                current_scheduler.load_state_dict(state['scheduler_state_dict'])
            except Exception as e:
                if is_main_process():
                    train_logger.info(f"Warning: Could not load scheduler state dict ({e}).")

            if is_main_process():
                train_logger.info('Resume from Epoch %d, global step %d, best validation loss: %.6f' %
                                  (start_epoch, global_step, best_val_loss))

    if is_main_process():
        tmp_test = validate_multi_qp(
                    model, val_loader, criterion, device, config, global_step
                )
        print(tmp_test)
        train_logger.info('========== Start Training ==========')
    
    for epoch in range(start_epoch, config.train.epochs + 1):
        epoch_start_time = time.time()

        if is_distributed:
            train_sampler.set_epoch(epoch)

        train_metrics, global_step, val_metrics_list, val_multi_qp_list  = \
                train_one_epoch(
                    model, 
                    disc_model,
                    criterion, 
                    optimizer, 
                    disc_optimizer,
                    train_loader, 
                    val_loader,
                    config, 
                    epoch,
                    scheduler=current_scheduler,
                    step_interval=step_interval,
                    global_step=global_step,
                    val_multi_qp_interval=val_multi_qp_interval,
                )
        train_loss, train_bpp, train_mse, train_d_loss = train_metrics['loss'], train_metrics['bpp'], train_metrics['mse'], train_metrics['d_loss']
        train_perc, train_adv = train_metrics['perc'], train_metrics['adv']
        
        # 处理本轮的验证结果
        if val_metrics_list:
            for val_metrics in val_metrics_list:
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    patience_counter = 0
                    if is_main_process():
                        train_logger.info('  *** New best validation loss: %.6f ***' % best_val_loss)
                        save_checkpoint(model, disc_model, optimizer, disc_optimizer, current_scheduler, epoch, global_step,
                                        best_val_loss, history, config, bias=bias, is_best=True)
                else:
                    patience_counter += 1
                    if is_main_process():
                        train_logger.info('  No improvement for %d validation cycles' % patience_counter)

                if patience_counter >= early_stop_patience:
                    if is_main_process():
                        train_logger.info('  *** Early stopping triggered after epoch %d ***' % epoch)
                    early_stop_triggered = True

        # 记录 history，保证各列表长度一致
        history['train_loss'].append(train_loss)
        history['train_bpp'].append(train_bpp)
        history['train_mse'].append(train_mse)
        history['train_d_loss'].append(train_d_loss)
        history['train_perc'].append(train_perc)
        history['train_adv'].append(train_adv)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # 记录本 epoch 的多 QP 验证结果
        if val_multi_qp_list:
            history['val_per_qp'].extend(val_multi_qp_list)

        if val_metrics_list:
            last_val = val_metrics_list[-1]
            history['val_loss'].append(last_val['loss'])
            history['val_bpp'].append(last_val['bpp'])
            history['val_psnr'].append(last_val['psnr'])
            history['val_ssim'].append(last_val['ssim'])
            history['val_lpips'].append(last_val['lpips']) 
            history['val_mse'].append(last_val.get('mse', 0.0))
        else:
            last_val_loss = history['val_loss'][-1] if history['val_loss'] else float('inf')
            last_val_bpp = history['val_bpp'][-1] if history['val_bpp'] else 0.0
            last_val_psnr = history['val_psnr'][-1] if history['val_psnr'] else 0.0
            last_val_ssim = history['val_ssim'][-1] if history['val_ssim'] else 0.0
            last_val_lpips = history['val_lpips'][-1] if history.get('val_lpips') and history['val_lpips'] else 1.0
            history['val_loss'].append(last_val_loss)
            history['val_bpp'].append(last_val_bpp)
            history['val_psnr'].append(last_val_psnr)
            history['val_ssim'].append(last_val_ssim)
            history['val_lpips'].append(last_val_lpips)
            history['val_mse'].append(0.0)

        epoch_time = time.time() - epoch_start_time

        train_logger.info('Epoch %03d/%d | Time: %.1fs' % (epoch, config.train.epochs, epoch_time))
        train_logger.info('\tTrain Loss: %.4f | BPP: %.4f | MSE: %.6f | Perceptual: %.4f | Adv: %.4f' % (train_loss, train_bpp, train_mse, train_perc, train_adv))
        
        if val_metrics_list:
            last_val = val_metrics_list[-1]
            train_logger.info('\tVal   Loss: %.4f | BPP: %.4f | PSNR: %.2f dB | SSIM: %.4f | LPIPS: %.4f' %
                             (last_val['loss'], last_val['bpp'], last_val['psnr'], last_val['ssim'], last_val['lpips']))
        else:
            train_logger.info('\tVal   Loss: %.4f | BPP: %.4f | PSNR: %.2f dB | SSIM: %.4f | LPIPS: %.4f' %
                           (history['val_loss'][-1], history['val_bpp'][-1], history['val_psnr'][-1], history['val_ssim'][-1], history['val_lpips'][-1]))
        
        train_logger.info('  LR: %.6f' % optimizer.param_groups[0]['lr'])

        # 保存最新训练状态（模型和训练状态分开保存）
        save_checkpoint(model, disc_model, optimizer, disc_optimizer, current_scheduler, epoch, global_step, best_val_loss, history, config, bias=bias,
                        is_best=False)

        # Warmup 结束后切换为 ReduceLROnPlateau  NOTE: move to train_one_epoch 
        if not switched_to_reduce_lr and epoch >= config.train.warmup_epochs:
            train_logger.info('  *** Switching to ReduceLROnPlateau scheduler ***')
            current_scheduler = reduce_lr_scheduler
            switched_to_reduce_lr = True
        
        # if early_stop_triggered:
        #     train_logger.info('  *** Early stopping: breaking training loop ***')
        #     break

        torch.cuda.empty_cache()

    cleanup_distributed()

    train_logger.info('========== Training Complete ==========')

    # 训练结束后绘图
    if is_main_process():
        plot_history(history, config.save_dir)


def plot_history(history, save_dir):
    """训练结束后绘制 step-metric 曲线并保存到 save_dir。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    # ---- 1. 训练曲线（按 epoch） ----
    epochs = list(range(1, len(history['train_loss']) + 1))
    if len(epochs) > 0:
        fig, axes = plt.subplots(4, 2, figsize=(14, 20))
        axes = axes.flatten()

        axes[0].plot(epochs, history['train_loss'], 'b-', linewidth=1)
        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss'); axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, history['train_bpp'], 'g-', linewidth=1)
        axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('BPP')
        axes[1].set_title('Training BPP'); axes[1].grid(True, alpha=0.3)

        axes[2].plot(epochs, history['train_mse'], 'r-', linewidth=1)
        axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('MSE')
        axes[2].set_title('Training MSE'); axes[2].grid(True, alpha=0.3)

        axes[3].plot(epochs, history['lr'], 'm-', linewidth=1)
        axes[3].set_xlabel('Epoch'); axes[3].set_ylabel('LR')
        axes[3].set_title('Learning Rate'); axes[3].grid(True, alpha=0.3)
        
        axes[4].plot(epochs, history['train_perc'], 'c-', linewidth=1)
        axes[4].set_xlabel('Epoch'); axes[4].set_ylabel('Perc Loss')
        axes[4].set_title('Training Perceptual Loss'); axes[4].grid(True, alpha=0.3)

        axes[5].plot(epochs, history['train_adv'], 'y-', linewidth=1)
        axes[5].set_xlabel('Epoch'); axes[5].set_ylabel('Adv Loss')
        axes[5].set_title('Training Adversarial Loss'); axes[5].grid(True, alpha=0.3)

        axes[6].plot(epochs, history['train_d_loss'], 'k-', linewidth=1)
        axes[6].set_xlabel('Epoch'); axes[6].set_ylabel('D Loss')
        axes[6].set_title('Training Discriminator Loss'); axes[6].grid(True, alpha=0.3)
        try:
            axes[3].set_yscale('log')
        except ValueError:
            pass
        
        fig.tight_layout()
        fig.suptitle('Training Metrics over Epochs', fontsize=14, fontweight='bold')
        fig.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        train_logger.info('Saved training curves plot to: ' + os.path.join(save_dir, 'training_curves.png'))

    # ---- 2. QP=32 验证曲线（按验证事件） ----
    if len(history['val_loss']) > 0:
        val_x = list(range(1, len(history['val_loss']) + 1))
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        axes[0].plot(val_x, history['val_loss'], 'b-o', markersize=3, linewidth=1)
        axes[0].set_xlabel('Validation Event'); axes[0].set_ylabel('Loss')
        axes[0].set_title('Validation Loss (Single QP)'); axes[0].grid(True, alpha=0.3)

        axes[1].plot(val_x, history['val_bpp'], 'g-o', markersize=3, linewidth=1)
        axes[1].set_xlabel('Validation Event'); axes[1].set_ylabel('BPP')
        axes[1].set_title('Validation BPP (Single QP)'); axes[1].grid(True, alpha=0.3)

        axes[2].plot(val_x, history['val_psnr'], 'c-o', markersize=3, linewidth=1)
        axes[2].set_xlabel('Validation Event'); axes[2].set_ylabel('PSNR (dB)')
        axes[2].set_title('Validation PSNR (Single QP)'); axes[2].grid(True, alpha=0.3)

        axes[3].plot(val_x, history['val_ssim'], 'm-o', markersize=3, linewidth=1)
        axes[3].set_xlabel('Validation Event'); axes[3].set_ylabel('SSIM')
        axes[3].set_title('Validation SSIM (Single QP)'); axes[3].grid(True, alpha=0.3)

        axes[4].plot(val_x, history['val_lpips'], 'k-o', markersize=3, linewidth=1)
        axes[4].set_xlabel('Validation Event'); axes[4].set_ylabel('LPIPS')
        axes[4].set_title('Validation LPIPS (Single QP)'); axes[4].grid(True, alpha=0.3)

        # 第 6 个子图可以留空或画 val_mse：
        axes[5].plot(val_x, history['val_mse'], 'orange', marker='o', markersize=3, linewidth=1)
        axes[5].set_xlabel('Validation Event'); axes[5].set_ylabel('MSE')
        axes[5].set_title('Validation MSE (Single QP)'); axes[5].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.suptitle('Validation Metrics (Single QP) over Validation Events', fontsize=14, fontweight='bold')
        fig.savefig(os.path.join(save_dir, 'validation_single_qp_curves.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        train_logger.info('Saved single QP validation curves plot to: ' + os.path.join(save_dir, 'validation_single_qp_curves.png'))

    # ---- 3. 多 QP 验证曲线（per-QP 指标随 step 变化） ----
    val_per_qp = history.get('val_per_qp', [])
    if len(val_per_qp) > 0:
        # 收集所有 QP 和 step
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

        # 提取每个 QP 的 bpp/psnr/ssim 序列
        qp_bpp = {qp: [] for qp in all_qps}
        qp_psnr = {qp: [] for qp in all_qps}
        qp_ssim = {qp: [] for qp in all_qps}
        for idx in sort_idx:
            entry = val_per_qp[idx]
            for qp in all_qps:
                if qp in entry['results']:
                    qp_bpp[qp].append(entry['results'][qp]['bpp'])
                    qp_psnr[qp].append(entry['results'][qp]['psnr'])
                    qp_ssim[qp].append(entry['results'][qp]['ssim'])
                else:
                    qp_bpp[qp].append(np.nan)
                    qp_psnr[qp].append(np.nan)
                    qp_ssim[qp].append(np.nan)

        # 3a. Per-QP PSNR over steps
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        cmap = plt.get_cmap('viridis')
        colors = [cmap(i / max(len(all_qps) - 1, 1)) for i in range(len(all_qps))]

        for i, qp in enumerate(all_qps):
            valid = ~np.isnan(qp_psnr[qp])
            if valid.sum() > 0:
                axes[0].plot(steps[valid], np.array(qp_psnr[qp])[valid],
                           'o-', color=colors[i], markersize=3, linewidth=1, label=f'QP{qp}')
        axes[0].set_xlabel('Step'); axes[0].set_ylabel('PSNR (dB)')
        axes[0].set_title('Per-QP PSNR over Steps'); axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=7, ncol=2)

        # 3b. Per-QP BPP over steps
        for i, qp in enumerate(all_qps):
            valid = ~np.isnan(qp_bpp[qp])
            if valid.sum() > 0:
                axes[1].plot(steps[valid], np.array(qp_bpp[qp])[valid],
                           'o-', color=colors[i], markersize=3, linewidth=1, label=f'QP{qp}')
        axes[1].set_xlabel('Step'); axes[1].set_ylabel('BPP')
        axes[1].set_title('Per-QP BPP over Steps'); axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=7, ncol=2)

        # 3c. Per-QP SSIM over steps
        for i, qp in enumerate(all_qps):
            valid = ~np.isnan(qp_ssim[qp])
            if valid.sum() > 0:
                axes[2].plot(steps[valid], np.array(qp_ssim[qp])[valid],
                           'o-', color=colors[i], markersize=3, linewidth=1, label=f'QP{qp}')
        axes[2].set_xlabel('Step'); axes[2].set_ylabel('SSIM')
        axes[2].set_title('Per-QP SSIM over Steps'); axes[2].grid(True, alpha=0.3)
        axes[2].legend(fontsize=7, ncol=2)

        fig.tight_layout()
        fig.suptitle('Per-QP Validation Metrics over Training Steps', fontsize=14, fontweight='bold')
        fig.savefig(os.path.join(save_dir, 'per_qp_metrics_over_steps.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        train_logger.info('Saved per-QP metrics plot to: ' + os.path.join(save_dir, 'per_qp_metrics_over_steps.png'))

        # ---- 4. RD 曲线（每个验证事件的 BPP vs PSNR） ----
        fig, ax = plt.subplots(figsize=(10, 8))
        for i, idx in enumerate(sort_idx):
            entry = val_per_qp[idx]
            qps_for_rd = sorted(entry['results'].keys())
            bpps = [entry['results'][q]['bpp'] for q in qps_for_rd]
            psnrs = [entry['results'][q]['psnr'] for q in qps_for_rd]
            color = plt.get_cmap('plasma')(i / max(len(sort_idx) - 1, 1))
            line, = ax.plot(bpps, psnrs, 'o-', color=color, markersize=4, linewidth=1,
                           alpha=0.7, label=f'Step {entry["step"]}')
            # 标注 QP
            for qp, bx, by in zip(qps_for_rd, bpps, psnrs):
                if i == len(sort_idx) - 1:  # 只标注最后一次
                    ax.annotate(f'QP{qp}', (bx, by), textcoords="offset points",
                               xytext=(5, 5), fontsize=7, alpha=0.8)

        ax.set_xlabel('BPP'); ax.set_ylabel('PSNR (dB)')
        ax.set_title('Rate-Distortion Curves over Training Steps', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, ncol=2, loc='lower right')
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, 'rd_curves.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        train_logger.info('Saved RD curves plot to: ' + os.path.join(save_dir, 'rd_curves.png'))

        # ---- 5. 最终 RD 曲线（最后一次多 QP 验证） ----
        last_entry = val_per_qp[sort_idx[-1]]
        qps_for_final = sorted(last_entry['results'].keys())
        bpps_final = [last_entry['results'][q]['bpp'] for q in qps_for_final]
        psnrs_final = [last_entry['results'][q]['psnr'] for q in qps_for_final]
        ssims_final = [last_entry['results'][q]['ssim'] for q in qps_for_final]

        fig, ax1 = plt.subplots(figsize=(8, 6))
        ax2 = ax1.twinx()

        line1, = ax1.plot(bpps_final, psnrs_final, 'b-o', markersize=6, linewidth=2, label='PSNR')
        line2, = ax2.plot(bpps_final, ssims_final, 'r-s', markersize=6, linewidth=2, label='SSIM')

        for qp, bx, by_psnr, by_ssim in zip(qps_for_final, bpps_final, psnrs_final, ssims_final):
            ax1.annotate(f'QP{qp}', (bx, by_psnr), textcoords="offset points",
                        xytext=(8, 6), fontsize=8, color='blue')
            ax2.annotate(f'QP{qp}', (bx, by_ssim), textcoords="offset points",
                        xytext=(8, -12), fontsize=8, color='red')

        ax1.set_xlabel('BPP'); ax1.set_ylabel('PSNR (dB)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax2.set_ylabel('SSIM', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax1.set_title(f'Final Rate-Distortion Curve (Step {last_entry["step"]})', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        lines = [line1, line2]
        ax1.legend(lines, [l.get_label() for l in lines], loc='lower right')
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, 'final_rd_curve.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        train_logger.info('Saved final RD curve plot to: ' + os.path.join(save_dir, 'final_rd_curve.png'))


if __name__ == '__main__':
    main()


