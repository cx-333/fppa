# -*- encoding: utf-8 -*-
'''
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.

@File    :   ema.py
@Time    :   2026/03/19 16:41:58
@Author  :   XinChen 
'''

import torch
import torch.nn as nn



class EMAScheme:
    def __init__(self, model: nn.Module, decay: float = 0.99):
        """
        decay: 衰减率，通常设为 0.99。
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # 初始化：克隆当前模型的所有可训练参数
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """
        更新公式：W_ema = (1 - decay) * W_current + decay * W_ema
        在 optimizer.step() 后调用。
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    assert name in self.shadow
                    # 确保 shadow 与当前参数在同一设备上
                    if self.shadow[name].device != param.device:
                        self.shadow[name] = self.shadow[name].to(param.device)
                    # 平滑更新
                    new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                    self.shadow[name].copy_(new_average)

    def apply_shadow(self):
        """
        将 EMA 权重应用到模型中（用于验证或保存）。
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                # 确保 shadow 与当前参数在同一设备上
                if self.shadow[name].device != param.device:
                    self.shadow[name] = self.shadow[name].to(param.device)
                param.data.copy_(self.shadow[name])

    def restore(self):
        """
        恢复模型原始权重（继续训练）。
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # 确保 backup 与当前参数在同一设备上
                if self.backup[name].device != param.device:
                    self.backup[name] = self.backup[name].to(param.device)
                param.data.copy_(self.backup[name])
        self.backup = {}
        

"""
使用方法：
    # 初始化模型、优化器和 EMA
    model = YourNVCModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    ema = EMAScheme(model, decay=0.999)

    for epoch in range(epochs):
        model.train()
        for x_t, x_prev in dataloader:
            optimizer.zero_grad()
            
            # 前向传播与损失计算 (R + lambda * D) [cite: 182]
            loss = model(x_t, x_prev)
            loss.backward()
            
            optimizer.step()
            
            # 关键步骤：更新 EMA 权重
            ema.update()

        # 验证阶段
        ema.apply_shadow() # 切换到平滑权重进行评估
        validate(model)
        ema.restore()      # 换回原始权重继续训练
"""
