# -*- coding: utf-8 -*-
"""1D-CNN 骨干网络（4 层卷积 + 2 层全连接，参数量 < 50K）。

设计约束（来自项目文档）
------------------------
- 4 层卷积 + 2 层全连接；
- **参数量必须 < 50K** —— 这是边缘部署（ESP32-S3）叙事成立的前提，
  也是"频率软约束带来的收益不是靠堆参数换来的"这一论点的支撑，故在测试里硬断言。

结构
----
输入 ``(B, 1, L)``，默认 L = 2048（12 kHz 下约 170 ms）::

    Conv1d(1  -> 16, k=15, s=2, p=7)  -> BN -> ReLU -> MaxPool(2)     L/4
    Conv1d(16 -> 32, k=7,  s=2, p=3)  -> BN -> ReLU -> MaxPool(2)     L/16
    Conv1d(32 -> 64, k=5,  s=2, p=2)  -> BN -> ReLU -> MaxPool(2)     L/64
    Conv1d(64 -> 96, k=3,  s=2, p=1)  -> BN -> ReLU -> MaxPool(2)     L/256
    [GAP ‖ GMP]  ->  192 维
    FC1(192 -> 64) -> ReLU -> Dropout                                 <-- embedding
    FC2(64  -> n_way)                                                 <-- logits

两级全局池化（GAP ‖ GMP）的理由：轴承冲击是**稀疏瞬态**，
全局平均会把它摊平，全局最大能保住"最强冲击"这一关键证据；
拼接两者兼顾能量与峰值。参数量代价为 0。

四个 baseline 共用同一个骨干
---------------------------
``CNN-CE`` / ``CNN-Freq`` 用 ``logits``；``ProtoNet`` / ``MatchingNet`` / ``FineTune``
用 ``embed`` 输出的 64 维特征做度量。**骨干完全相同**，只有头与损失不同
—— 这是对比实验公平性的基础（论文 6.1 节要写明）。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, k: int, stride: int = 1, pool: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Conv1d(cin, cout, kernel_size=k, stride=stride,
                              padding=k // 2, bias=True)
        self.bn = nn.BatchNorm1d(cout)
        self.pool = nn.MaxPool1d(pool) if pool and pool > 1 else nn.Identity()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn(self.conv(x))))
        return self.drop(x)


class BearingCNN(nn.Module):
    """1D-CNN 骨干 + 两级分类头。

    Parameters
    ----------
    in_channels : 输入通道，单轴振动 = 1
    channels    : 4 层卷积输出通道
    kernels     : 4 层卷积核长
    strides     : 4 层卷积步长
    pools       : 4 层池化核长
    fc_hidden   : FC1 宽度（= embedding 维度）
    n_way       : 分类数（4-way）
    pool_mode   : 'gap_gmp' | 'gap' | 'gmp'
    """

    def __init__(self, in_channels: int = 1, channels: Sequence[int] = (16, 32, 64, 96),
                 kernels: Sequence[int] = (15, 7, 5, 3), strides: Sequence[int] = (2, 2, 2, 2),
                 pools: Sequence[int] = (2, 2, 2, 2), fc_hidden: int = 64,
                 n_way: int = 4, dropout: float = 0.1,
                 pool_mode: str = 'gap_gmp'):
        super().__init__()
        assert len(channels) == len(kernels) == len(strides) == len(pools) == 4, \
            '必须恰好 4 层卷积'
        assert pool_mode in ('gap_gmp', 'gap', 'gmp')
        self.cfg = dict(in_channels=in_channels, channels=list(channels),
                        kernels=list(kernels), strides=list(strides), pools=list(pools),
                        fc_hidden=fc_hidden, n_way=n_way, dropout=dropout,
                        pool_mode=pool_mode)

        blocks = []
        cin = in_channels
        for i, (co, k, s, p) in enumerate(zip(channels, kernels, strides, pools)):
            blocks.append(ConvBlock(cin, co, k, s, p, dropout=0.0))
            cin = co
        self.blocks = nn.Sequential(*blocks)
        self.out_channels = int(cin)
        self.pool_mode = pool_mode
        feat_dim = self.out_channels * (2 if pool_mode == 'gap_gmp' else 1)
        self.feat_dim = feat_dim

        self.fc1 = nn.Linear(feat_dim, fc_hidden)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_hidden, n_way)
        self.embed_dim = int(fc_hidden)

    # ------------------------------------------------------------------
    def pool_features(self, h: torch.Tensor) -> torch.Tensor:
        if self.pool_mode == 'gap_gmp':
            return torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=1)
        if self.pool_mode == 'gap':
            return h.mean(dim=-1)
        return h.amax(dim=-1)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """卷积特征（池化后），供可视化/复杂度分析使用。"""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.pool_features(self.blocks(x))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """embedding（FC1 输出，ReLU 后），供 ProtoNet / MatchingNet 使用。"""
        f = self.features(x)
        return F.relu(self.fc1(f))

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        e = self.embed(x)
        if return_embedding:
            return e
        return self.fc2(self.drop(e))

    # ------------------------------------------------------------------
    @classmethod
    def from_cfg(cls, cfg: dict) -> 'BearingCNN':
        m = cfg['model']
        return cls(in_channels=int(m['in_channels']), channels=tuple(m['channels']),
                   kernels=tuple(m['kernels']), strides=tuple(m['strides']),
                   pools=tuple(m.get('pools', [2, 2, 2, 2])), fc_hidden=int(m['fc_hidden']),
                   n_way=int(m['n_way']), dropout=float(m.get('dropout', 0.1)),
                   pool_mode=str(m.get('pool_mode', 'gap_gmp')))


# ----------------------------------------------------------------------
# 统计工具
# ----------------------------------------------------------------------
def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    ps = model.parameters()
    return sum(p.numel() for p in ps if (p.requires_grad or not trainable_only))


def parameter_breakdown(model: nn.Module) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv1d, nn.Linear, nn.BatchNorm1d)):
            n = sum(p.numel() for p in mod.parameters())
            if n:
                out[name] = n
    out['TOTAL'] = count_parameters(model)
    return out


def estimate_flops(model: BearingCNN, length: int = 2048) -> Dict[str, float]:
    """估算单样本前向 MACs / FLOPs。

    卷积层 : MACs = L_out * C_out * C_in * K
    全连接 : MACs = in * out
    FLOPs ≈ 2 × MACs（一次乘 + 一次加）。
    """ 
    macs = 0.0
    L = length
    cin = model.cfg['in_channels']
    for co, k, s, p in zip(model.cfg['channels'], model.cfg['kernels'],
                           model.cfg['strides'], model.cfg['pools']):
        L_conv = L // s
        macs += L_conv * co * cin * k
        L = L_conv // p
        cin = co
    feat = model.feat_dim
    macs += feat * model.cfg['fc_hidden']
    macs += model.cfg['fc_hidden'] * model.cfg['n_way']
    return {'input_length': length, 'macs': macs, 'flops': 2.0 * macs,
            'macs_m': macs / 1e6, 'flops_m': 2.0 * macs / 1e6}


def assert_param_budget(model: nn.Module, max_params: int = 50000) -> int:
    n = count_parameters(model)
    if n >= max_params:
        raise AssertionError('参数量 %d 超过上限 %d' % (n, max_params))
    return n


# ----------------------------------------------------------------------
if __name__ == '__main__':
    torch.manual_seed(0)
    m = BearingCNN()
    n = count_parameters(m)
    print('=== BearingCNN 结构 ===')
    print(m)
    print('\n=== 参数量分解 ===')
    for k, v in parameter_breakdown(m).items():
        print('  %-16s %8d' % (k, v))
    print('\n总参数量 = %d  (上限 50000)  ->  %s' % (n, 'PASS' if n < 50000 else 'FAIL'))
    print('embedding 维度 = %d' % m.embed_dim)

    x = torch.randn(8, 1, 2048)
    logits = m(x)
    emb = m(x, return_embedding=True)
    print('\n输入 %s -> logits %s, embed %s' % (tuple(x.shape), tuple(logits.shape), tuple(emb.shape)))
    assert logits.shape == (8, 4)
    assert emb.shape == (8, m.embed_dim)

    # 不同长度也应能跑
    for L in (1024, 4096):
        y = m(torch.randn(2, 1, L))
        print('  L=%-5d -> %s' % (L, tuple(y.shape)))

    fl = estimate_flops(m, 2048)
    print('\n=== 计算量估算（单样本, L=2048）===')
    print('  MACs  = %.3f M' % fl['macs_m'])
    print('  FLOPs = %.3f M' % fl['flops_m'])
    print('INT8 权重量 ≈ %.1f KB (参数量 x 1 byte)' % (n / 1024.0))
