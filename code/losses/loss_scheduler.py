# -*- coding: utf-8 -*-
"""频率软约束权重 λ(t) 的调度策略（对应论文 5.6 节）。

三种策略
--------
**stepwise（默认，论文主用）**
    前 ``warmup_epochs`` 轮 λ=0，之后每 ``step`` 轮增加 ``step_value``，封顶 ``lambda_max``::

        λ(t) = 0                                       t < warmup
        λ(t) = min(λ_max, step_value * (1 + (t-warmup)//step))   否则

    默认 (warmup=20, step=10, step_value=0.1, λ_max=0.5) 得到：
    t<20 → 0；20-29 → 0.1；30-39 → 0.2；40-49 → 0.3；50-59 → 0.4；≥60 → 0.5

**fixed**
    λ 恒为 ``fixed_lambda``，用作"是否需要动态调度"的消融对照。

**adaptive**
    λ 随训练进度光滑上升（sigmoid 斜坡），避免 stepwise 的台阶在切换瞬间
    引起损失突跳::

        λ(t) = λ_max * sigmoid((t - warmup) / step)

动机（为什么必须 warmup）
-------------------------
训练初期网络输出接近均匀分布，``p_{i,c} ≈ 1/4``，此时 L_freq 的梯度方向
几乎与真实类别无关（相当于对四个类别都施加等强度约束），
会与 CE 的梯度对抗、把特征空间搞乱。
先让 CE 把模型带到"大致可分"的状态（前 20 轮），再逐步引入物理约束，
是 stepwise 策略有效的直接原因 —— 这一点在消融实验中会被验证
（λ=0 与 λ 直接取 0.5 无 warmup 的对照）。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence


class LambdaScheduler:
    """λ(t) 调度器基类。``t`` 为 0-based epoch。"""

    name = 'base'

    def value(self, epoch: int) -> float:
        raise NotImplementedError

    def values(self, n_epochs: int) -> List[float]:
        return [self.value(t) for t in range(int(n_epochs))]

    def curve(self, n_epochs: int) -> Dict[str, List[float]]:
        return {'epoch': list(range(int(n_epochs))), 'lambda': self.values(n_epochs)}

    def describe(self) -> dict:
        return {'name': self.name}


class StepwiseLambda(LambdaScheduler):
    name = 'stepwise'

    def __init__(self, warmup_epochs: int = 20, step: int = 10,
                 step_value: float = 0.1, lambda_max: float = 0.5):
        if step <= 0:
            raise ValueError('step 必须 > 0')
        self.warmup_epochs = int(warmup_epochs)
        self.step = int(step)
        self.step_value = float(step_value)
        self.lambda_max = float(lambda_max)

    def value(self, epoch: int) -> float:
        t = int(epoch)
        if t < self.warmup_epochs:
            return 0.0
        k = (t - self.warmup_epochs) // self.step
        # round 到 12 位：step_value*(1+k) 在浮点下会出现 0.1*3 = 0.30000000000000004，
        # 会让 λ 序列与文档规定的 0.1/0.2/0.3... 严格比对失败（且污染日志可读性）
        return float(min(self.lambda_max, round(self.step_value * (1 + k), 12)))

    def describe(self) -> dict:
        return {'name': self.name, 'warmup_epochs': self.warmup_epochs, 'step': self.step,
                'step_value': self.step_value, 'lambda_max': self.lambda_max}


class FixedLambda(LambdaScheduler):
    name = 'fixed'

    def __init__(self, fixed_lambda: float = 0.3):
        self.fixed_lambda = float(fixed_lambda)

    def value(self, epoch: int) -> float:
        return self.fixed_lambda

    def describe(self) -> dict:
        return {'name': self.name, 'fixed_lambda': self.fixed_lambda}


class AdaptiveLambda(LambdaScheduler):
    name = 'adaptive'

    def __init__(self, warmup_epochs: int = 20, step: int = 10, lambda_max: float = 0.5):
        self.warmup_epochs = int(warmup_epochs)
        self.step = int(step)
        self.lambda_max = float(lambda_max)

    def value(self, epoch: int) -> float:
        t = int(epoch)
        z = (t - self.warmup_epochs) / float(self.step)
        # 数值稳定的 sigmoid
        if z >= 0:
            s = 1.0 / (1.0 + math.exp(-z))
        else:
            e = math.exp(z)
            s = e / (1.0 + e)
        return float(self.lambda_max * s)

    def describe(self) -> dict:
        return {'name': self.name, 'warmup_epochs': self.warmup_epochs,
                'step': self.step, 'lambda_max': self.lambda_max}


class ConstantLambda(LambdaScheduler):
    """λ ≡ 0，即纯 CE 基线（CNN-CE）。"""

    name = 'zero'

    def value(self, epoch: int) -> float:
        return 0.0


def build_lambda_scheduler(loss_cfg: dict) -> LambdaScheduler:
    """按配置构造调度器。"""
    name = str(loss_cfg.get('lambda_strategy', 'stepwise')).lower()
    if name == 'stepwise':
        return StepwiseLambda(loss_cfg.get('warmup_epochs', 20), loss_cfg.get('step', 10),
                              loss_cfg.get('step_value', 0.1), loss_cfg.get('lambda_max', 0.5))
    if name == 'fixed':
        return FixedLambda(loss_cfg.get('fixed_lambda', 0.3))
    if name == 'adaptive':
        return AdaptiveLambda(loss_cfg.get('warmup_epochs', 20), loss_cfg.get('step', 10),
                              loss_cfg.get('lambda_max', 0.5))
    if name in ('zero', 'none', 'off'):
        return ConstantLambda()
    raise ValueError('未知 lambda_strategy: %s' % name)


if __name__ == '__main__':
    print('=== stepwise（默认参数）λ(t) ===')
    for sched in [StepwiseLambda(20, 10, 0.1, 0.5), FixedLambda(0.3), AdaptiveLambda(20, 10, 0.5)]:
        v = sched.values(80)
        print('\n%s: %s' % (sched.describe(), ['%.2f' % x for x in v]))
        if sched.name == 'stepwise':
            # 断言：严格符合"前20轮0，之后每10轮+0.1至0.5"
            expect = ([0.0] * 20 + [0.1] * 10 + [0.2] * 10 + [0.3] * 10 + [0.4] * 10 + [0.5] * 20)
            print('  与文档规定序列一致: %s' % ('PASS' if v == expect else 'FAIL'))
            assert v == expect
    print('\n=== λ=0 消融（ConstantLambda）===')
    print(set(ConstantLambda().values(50)))

    print('\n=== 无 warmup 反例（λ 直接从 0.5 起步）===')
    bad = StepwiseLambda(warmup_epochs=0, step=1, step_value=0.5, lambda_max=0.5)
    print('前5轮 λ =', ['%.2f' % x for x in bad.values(5)], '(说明 warmup 的必要性)')
