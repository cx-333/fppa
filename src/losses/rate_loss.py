from typing import Optional, Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class RateLoss(nn.Module):
    def __init__(self, loss_weight: float = 1.0):
        super().__init__()
        self.lamb_rate = loss_weight

    def forward(self, bpp, **kwargs):
        return self.lamb_rate * bpp     # （B， ）


@LOSS_REGISTRY.register()
class HificRateLoss(nn.Module):
    def __init__(
        self,
        lambda_A: float,
        lambda_B: float,
        target_rate: float,
        lambda_schedule: Optional[Dict] = None,
        target_rate_schedule: Optional[Dict[str, List]] = None,
    ) -> None:
        """Dynamic rate loss used in HiFiC

        Args:
            lambda_A (float): applied when rate is higher than target
            lambda_B (float): applied when rate is lower than target
            target_rate (float):
            lambda_schedule (Optional[Dict], optional): 'steps': [List], 'vals': [List]
            target_rate_schedule (Optional[Dict], optional): 'steps': [List], 'vals': [List]
        """
        super().__init__()
        assert (
            lambda_A > lambda_B
        ), f"Expected lambda_A > lambda_B, got (A) {lambda_A} <= (B) {lambda_B}"
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B
        self.target_rate = target_rate

        self._check_schedule(lambda_schedule)
        self._check_schedule(target_rate_schedule)
        self.lambda_schedule = lambda_schedule
        self.target_rate_schedule = target_rate_schedule

    @staticmethod
    def _check_schedule(schedule: Optional[Dict[str, List]] = None) -> None:
        if schedule is None:
            return
        assert isinstance(
            schedule, dict
        ), f"schedule must be dict, but {type(schedule)}"
        assert "vals" in schedule, 'schedule must have "vals" as a key'
        assert "steps" in schedule, 'schedule must have "steps" as a key'
        vals = schedule["vals"]
        steps = schedule["steps"]
        assert isinstance(
            vals, list
        ), f'schedule["vals"] must be list, but {type(vals)}'
        assert isinstance(
            steps, list
        ), f'schedule["steps"] must be list, but {type(steps)}'
        assert (
            len(vals) == len(steps) + 1
        ), f"Requirement: len(vals) = len(steps)+1, but {len(vals)} vs {len(steps)}"

    @staticmethod
    def get_scheduled_params(
        param: float, param_schedule: Dict, step_counter: int
    ) -> float:
        vals, steps = param_schedule["vals"], param_schedule["steps"]
        idx = np.where(step_counter < np.array(steps + [step_counter + 1]))[0][0]
        param *= vals[idx]
        return param

    def forward(
        self, bpp: torch.Tensor, qbpp: torch.Tensor, current_iter: int, **kwargs
    ) -> torch.Tensor:
        lambda_A, lambda_B = self.lambda_A, self.lambda_B
        if self.lambda_schedule:
            lambda_A = self.get_scheduled_params(
                lambda_A, self.lambda_schedule, current_iter
            )
            lambda_B = self.get_scheduled_params(
                lambda_B, self.lambda_schedule, current_iter
            )

        target_bpp = self.target_rate
        if self.target_rate_schedule:
            target_bpp = self.get_scheduled_params(
                target_bpp, self.target_rate_schedule, current_iter
            )

        qbpp = torch.mean(qbpp.detach()).item()
        weight = (
            lambda_A if qbpp > target_bpp else lambda_B
        )
        return weight * torch.mean(bpp)


@LOSS_REGISTRY.register()
class HificVariableRateLoss(HificRateLoss):
    def __init__(
        self,
        lambda_A: List[float],
        lambda_B: Union[List[float], float],
        target_rate: List[float],
        lambda_schedule: Optional[Dict] = None,
        target_rate_schedule: Optional[Dict[str, List]] = None,
        device: str="cuda"
    ) -> None:
        # super(HificRateLoss, self).__init__()
        nn.Module.__init__(self)
        if isinstance(lambda_B, float):
            lambda_B = [lambda_B] * len(lambda_A)
        self.check_lambda_target(lambda_A, lambda_B, target_rate)
        self.lambda_A = torch.tensor(lambda_A, dtype=torch.float, device=device)
        self.lambda_B = torch.tensor(lambda_B, dtype=torch.float, device=device)
        self.target_rate = torch.tensor(target_rate, dtype=torch.float, device=device)

        self._check_schedule(lambda_schedule)
        self._check_schedule(target_rate_schedule)
        self.lambda_schedule = lambda_schedule
        self.target_rate_schedule = target_rate_schedule

    @staticmethod
    def check_lambda_target(lambda_A, lambda_B, target_rate):
        assert len(lambda_A) == len(lambda_B)
        assert len(lambda_A) == len(target_rate)

        target_rate_ = sorted(target_rate)
        assert target_rate == target_rate_
        lambda_A_ = sorted(lambda_A, reverse=True)
        assert lambda_A_ == lambda_A

        for i, (a, b) in enumerate(zip(lambda_A, lambda_B)):
            assert (
                a > b
            ), f"Expected lambda_A > lambda_B, got (A[{i}]) {a} <= (B[{i}]) {b}"

    def forward(
        self,
        bpp: torch.Tensor,
        qbpp: torch.Tensor,
        current_iter: int,
        rate_ind: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # is_multi = isinstance(rate_ind, torch.Tensor) and rate_ind.numel() > 1
        # device = bpp.device
        # dtype = bpp.dtype
        # bpp (B, )
        # if is_multi:
        #     rate_ind_l = rate_ind.long()
        #     lambda_A = torch.tensor(self.lambda_A, device=device, dtype=dtype)[rate_ind_l]
        #     lambda_B = torch.tensor(self.lambda_B, device=device, dtype=dtype)[rate_ind_l]
        #     target_bpp = torch.tensor(self.target_rate, device=device, dtype=dtype)[rate_ind_l]
        # else:
        #     _rate_ind = int(rate_ind.item() if isinstance(rate_ind, torch.Tensor) else rate_ind)
        #     lambda_A = self.lambda_A[_rate_ind]
        #     lambda_B = self.lambda_B[_rate_ind]
        #     target_bpp = self.target_rate[_rate_ind]
        lambda_A = self.lambda_A[rate_ind]
        lambda_B = self.lambda_B[rate_ind]
        target_bpp = self.target_rate[rate_ind]

        if self.lambda_schedule:
            scale = self.get_scheduled_params(1.0, self.lambda_schedule, current_iter)
            lambda_A = lambda_A * scale
            lambda_B = lambda_B * scale

        if self.target_rate_schedule:
            scale = self.get_scheduled_params(1.0, self.target_rate_schedule, current_iter)
            target_bpp = target_bpp * scale

        # qbpp 可能是 (B,) 或 (B, T)
        qbpp_mean = qbpp.detach().mean(dim=tuple(range(1, qbpp.dim()))) if qbpp.dim() > 1 else qbpp.detach()
        weight = torch.where(qbpp_mean > target_bpp, lambda_A, lambda_B)
        # bpp_mean = bpp.mean(dim=tuple(range(1, bpp.dim()))) if bpp.dim() > 1 else bpp
        return weight * bpp         # (B, )



