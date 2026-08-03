# -*- encoding: utf-8 -*-
'''
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.

@File    :   utils.py
@Time    :   2026/03/20 09:32:04
@Author  :   XinChen 
'''


import torch
import os 
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
import random
import numpy as np
import subprocess
import shutil
import torch.nn.functional as F 
from typing import Union, Tuple, Callable, Any
import logging
import json


def save_checkpoint(state, is_best, save_dir, filename="checkpoint.pth.tar"):
    """ 保存训练断点和最佳模型 """
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)
    if is_best:
        # torch.save(state, os.path.join(save_dir, "best_" + filename))
        shutil.copyfile(filepath, os.path.join(save_dir, "best_" + filename))
        
        
def get_state_dict(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=True)
    if "state_dict" in ckpt:
        ckpt = ckpt['state_dict']
    if "net" in ckpt:
        ckpt = ckpt["net"]
    if "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    consume_prefix_in_state_dict_if_present(ckpt, prefix="module.")
    return ckpt


def set_seed(seed=42):
    # 1. Python 原生随机库
    random.seed(seed)
    # 2. 环境变量（某些操作如数据增强可能用到）
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 3. NumPy
    np.random.seed(seed)
    # 4. PyTorch CPU
    torch.manual_seed(seed)
    # 5. PyTorch GPU (所有显卡)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    # 6. CuDNN 确定性设置 (关键！)
    # 设为 True 保证每次卷积算法选择一致，但会略微降低训练速度
    torch.backends.cudnn.deterministic = True
    # 设为 False 禁用自动寻找最快算法，确保结果可复现
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")
    
    
def start_tensorboard(log_dir, port=6006):
    """ 启动 TensorBoard 日志记录 """
    subprocess.run(f"cd {log_dir}", shell=True)
    subprocess.run(f"tensorboard --logdir {log_dir} --port {port}", shell=True)
    print(f"TensorBoard started, port: {port}")


def analysis_model(
    model: torch.nn.Module,
    input_size: Union[Tuple[int, ...], torch.Tensor, Tuple[torch.Tensor, ...]] = (1, 3, 256, 256),
    device: str = 'cpu',
    input_fn: Callable[[], Any] = None,
    **kwargs
) -> dict:
    """
    分析神经网络模型的统计信息
    
    Args:
        model: PyTorch 模型
        input_size: 输入尺寸或输入张量/元组
            - Tuple: 如 (1, 3, 256, 256)，自动创建随机张量
            - torch.Tensor: 直接作为输入
            - Tuple[torch.Tensor, ...]: 多个输入张量的元组
        device: 运行设备，默认 'cpu'
        input_fn: 自定义输入准备函数，返回输入张量或元组
                  优先级最高，若提供则忽略 input_size
        **kwargs: 传递给 input_fn 的额外参数
    example:
        def prepare_input():
            x = torch.randn(1, 3, 256, 256)
            qp = torch.tensor([32])  # 或 qp_scale 等
            return x, qp
        stats = analysis_model(model, input_fn=prepare_input)
    Returns:
        dict: 包含以下统计信息的字典
            - params (float): 参数量，单位 M (百万)
            - flops (float): FLOPs，单位 G (十亿)
            - macs (float): 乘加运算数
            - macs_per_pixel (float): 每像素 MACs，单位 k (千)
            - model_size (float): 模型大小，单位 M (MB)
    
    Examples:
        # 1. 简单图像模型
        stats = analysis_model(model, input_size=(1, 3, 256, 256))
        
        # 2. 需要额外参数 (如 qp)
        def prepare_input():
            x = torch.randn(1, 3, 256, 256)
            qp = torch.tensor([32])
            return x, qp
        stats = analysis_model(model, input_fn=prepare_input)
        
        # 3. 直接使用张量输入
        dummy_input = torch.randn(1, 3, 256, 256)
        stats = analysis_model(model, input_size=dummy_input)
    """
    model = model.to(device)
    model.eval()
    
    # ========== 准备输入数据 ==========
    if input_fn is not None:
        # 使用自定义输入函数
        dummy_input = input_fn(**kwargs)
        if isinstance(dummy_input, torch.Tensor):
            dummy_input = dummy_input.to(device)
            input_shape = tuple(dummy_input.shape)
        elif isinstance(dummy_input, (tuple, list)):
            dummy_input = tuple(x.to(device) if isinstance(x, torch.Tensor) else x for x in dummy_input)
            # 从第一个张量获取形状
            for x in dummy_input:
                if isinstance(x, torch.Tensor):
                    input_shape = tuple(x.shape)
                    break
        else:
            input_shape = input_size if isinstance(input_size, tuple) else (1, 3, 256, 256)
    elif isinstance(input_size, torch.Tensor):
        # 直接使用传入的张量
        dummy_input = input_size.to(device)
        input_shape = tuple(dummy_input.shape)
    elif isinstance(input_size, tuple) and len(input_size) > 0 and isinstance(input_size[0], torch.Tensor):
        # 元组形式的多个张量
        dummy_input = tuple(x.to(device) for x in input_size)
        for x in dummy_input:
            if isinstance(x, torch.Tensor):
                input_shape = tuple(x.shape)
                break
    else:
        # 从尺寸创建随机张量
        dummy_input = torch.randn(input_size).to(device)
        input_shape = input_size
    
    # 解析输入尺寸获取 H, W
    if len(input_shape) >= 4:
        batch_size, C, H, W = input_shape[0], input_shape[1], input_shape[2], input_shape[3]
    elif len(input_shape) == 3:
        C, H, W = input_shape[0], input_shape[1], input_shape[2]
        batch_size = 1
    else:
        H, W = 256, 256  # 默认值
        batch_size, C = 1, 3
    
    # ========== 1. 参数量统计 (M) ==========
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_m = total_params / 1e6
    
    # ========== 2. 模型大小 (MB) ==========
    # 计算参数占用的字节数 (假设 float32)
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    # 计算缓冲区大小 (running_mean, running_var等)
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
    model_size_mb = (param_size + buffer_size) / (1024 ** 2)
    
    # ========== 3. FLOPs 和 MACs 统计 ==========
    try:
        # 尝试使用 thop 库
        from thop import profile
        if isinstance(dummy_input, torch.Tensor):
            flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
        elif isinstance(dummy_input, (tuple, list)):
            flops, _ = profile(model, inputs=dummy_input, verbose=False)
        else:
            flops = 0
        macs = flops / 2  # FLOPs ≈ 2 * MACs
        flops_g = flops / 1e9
    except ImportError:
        # 如果 thop 不可用，使用自定义统计
        flops_g = _calculate_flops_custom(model, dummy_input, device)
        macs = flops_g * 1e9 / 2
    
    # ========== 4. MACs/pixel (k) ==========
    num_pixels = H * W
    macs_per_pixel_k = (macs / num_pixels) / 1e3
    
    # ========== 打印结果 ==========
    print("=" * 50)
    print("模型分析结果 Model Analysis")
    print("=" * 50)
    print(f"输入尺寸: {input_shape}")
    print(f"设备: {device}")
    print("-" * 50)
    print(f"参数量 Params:       {params_m:.4f} M")
    print(f"FLOPs:               {flops_g:.4f} G")
    print(f"MACs/pixel:          {macs_per_pixel_k:.4f} k")
    print(f"模型大小 Model Size: {model_size_mb:.4f} M")
    print("-" * 50)
    print(f"可训练参数:          {trainable_params / 1e6:.4f} M")
    print(f"总像素数:            {num_pixels} ({H}x{W})")
    print("=" * 50)
    
    return {
        'params': params_m,              # 参数量 (M)
        'flops': flops_g,                # FLOPs (G)
        'macs': macs,                    # MACs
        'macs_per_pixel': macs_per_pixel_k,  # MACs/pixel (k)
        'model_size': model_size_mb,     # 模型大小 (MB)
        'total_params': total_params,
        'trainable_params': trainable_params,
        'input_shape': input_shape,
    }


def _calculate_flops_custom(model, dummy_input, device):
    """
    自定义 FLOPs 计算方法（当 thop 不可用时使用）
    通过注册 hook 统计各层的乘加操作
    """
    flops_list = []
    
    def conv_hook(module, input, output):
        # 卷积层 FLOPs: 2 * Cin * Cout * K * K * Hout * Wout
        batch_size, in_channels, in_h, in_w = input[0].size()
        out_channels, _, kernel_h, kernel_w = module.weight.size()
        out_h, out_w = output.size(2), output.size(3)
        
        groups = module.groups
        conv_per_pos = in_channels * kernel_h * kernel_w
        
        flops = batch_size * out_channels * out_h * out_w * conv_per_pos // groups * 2
        flops_list.append(flops)
    
    def linear_hook(module, input, output):
        # 全连接层 FLOPs: 2 * Cin * Cout
        in_features = module.in_features
        out_features = module.out_features
        batch_size = input[0].size(0)
        flops = batch_size * in_features * out_features * 2
        flops_list.append(flops)
    
    def bn_hook(module, input, output):
        # BatchNorm FLOPs: 2 * C * H * W (缩放 + 偏移)
        batch_size, num_features, h, w = input[0].size()
        flops = batch_size * num_features * h * w * 2
        flops_list.append(flops)
    
    hooks = []
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            hooks.append(module.register_forward_hook(bn_hook))
    
    # 前向传播
    with torch.no_grad():
        if isinstance(dummy_input, torch.Tensor):
            model(dummy_input)
        elif isinstance(dummy_input, (tuple, list)):
            model(*dummy_input)
    
    # 移除 hooks
    for hook in hooks:
        hook.remove()
    
    total_flops = sum(flops_list)
    return total_flops / 1e9


def get_padding_size(height, width, p=64):
    new_h = (height + p - 1) // p * p
    new_w = (width + p - 1) // p * p
    padding_right = new_w - width
    padding_bottom = new_h - height
    return padding_bottom, padding_right


def replicated_pad(x, pad_b, pad_r):
    if pad_b == 0 and pad_r == 0:
        return x
    return F.pad(x, (0, pad_r, 0, pad_b), mode="replicate")


def unfreeze_model(model):
    module = {}
    for n, p in model.named_parameters():
        p.requires_grad = True
        module[n] = p.shape
    return module


def freeze_module(model):
    module = {}
    for n, p in model.named_parameters():
        p.requires_grad = False 
        module[n] = p.shape
        
    return module


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def logger_stepup(logpath, filepath, package_files=[]):
    formatter = logging.Formatter('%(asctime)s %(levelname)s - %(funcName)s: %(message)s', "%H:%M:%S")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    stream = logging.StreamHandler()
    stream.setLevel('INFO'.upper())
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    info_file_handler = logging.FileHandler(logpath, mode="a")
    info_file_handler.setLevel('INFO'.upper())
    info_file_handler.setFormatter(formatter)
    logger.addHandler(info_file_handler)

    logger.info(filepath)

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())
    return logger



class AverageMeter:
    """Compute running average."""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def reset(self):
        self.__init__()
        

def save_config(config):
    os.makedirs(config.train.save_dir, exist_ok=True)
    config_save_path = os.path.join(config.train.save_dir, 'config.json')
    with open(config_save_path, 'w') as f:
        json.dump(dict(config), f, indent=4, default=str)
    print('Config saved to: ' + config_save_path)
    