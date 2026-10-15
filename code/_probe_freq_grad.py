# -*- coding: utf-8 -*-
"""诊断：频率软约束 L_freq 到底有没有"说话的能力"？

起因（实测事实，不是猜测）
--------------------------
在 -8 dB、1-shot 设定下，CNN-CE 与 CNN-Freq 的结果**分毫不差**：
in-domain 0.8110 vs 0.8110，cross-2hp 0.7849 vs 0.7849，val_acc 0.771 vs 0.771。
0 dB 下同样如此（两者都是 1.000）。

必须判定原因。本脚本用真实 CWRU episode（含 -8 dB 加噪）量化四件事：

  A. ∂L_freq/∂logits 与 ∂L_CE/∂logits 的量级对比，随 logits 饱和程度变化；
  B. 只优化 L_freq（忽略 CE），logits 能否被推到"峰值匹配"的类；
  C. 控制变量：把物理先验 H 的列打乱（错误先验），L_freq 是否显著变化
     —— 若几乎不变，说明约束对 logits 根本不敏感；
  D. 物理先验本身的判别力：正确类的相对偏差 rel 是否普遍小于最好的错误类。

用法
----
    python _probe_freq_grad.py            # 默认 SNR=-8 dB
    python _probe_freq_grad.py --snr 0
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402
import torch.nn.functional as F                                 # noqa: E402
import yaml                                                     # noqa: E402

from data.cwru_loader import prepare, add_noise_to_pools        # noqa: E402
from data.envelope_spectrum import EnvelopeConfig               # noqa: E402
from losses.physics_loss import frequency_soft_constraint       # noqa: E402
from losses.physics_loss import FrequencySoftConstraintLoss     # noqa: E402
from training.episode_sampler import EpisodeSampler             # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load_cfg() -> dict:
    with open(os.path.join(HERE, 'configs', 'default.yaml'), encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg['freq']['shaft_rpm'] = {int(k): float(v) for k, v in cfg['freq']['shaft_rpm'].items()}
    for k in ('raw_dir', 'processed_dir'):
        cfg['data'][k] = os.path.join(ROOT, cfg['data'][k])
    for k, v in list(cfg['paths'].items()):
        cfg['paths'][k] = v if os.path.isabs(v) else os.path.join(ROOT, v)
    return cfg


def collect_peaks(pool, n_episodes: int, n_shot: int, seed: int = 0):
    """抽 n_episodes 个 episode，把 query 的 (peak, y) 拼起来，得到统计样本。"""
    s = EpisodeSampler(pool, 4, n_shot, 15, seed=seed, device='cpu', name='train')
    pks, ys = [], []
    for _ in range(n_episodes):
        ep = s.sample()
        pks.append(ep.query_peak)
        ys.append(ep.query_y)
    return torch.cat(pks, 0), torch.cat(ys, 0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--snr', type=float, default=-8.0, help='加噪 SNR(dB)')
    ap.add_argument('--episodes', type=int, default=50)
    ap.add_argument('--n-shot', type=int, default=1, dest='n_shot')
    args = ap.parse_args(argv)

    torch.manual_seed(0)
    cfg = load_cfg()
    pools = prepare(cfg, force=False, verbose=False)
    if args.snr is not None:
        pools = add_noise_to_pools(pools, args.snr, 12345,
                                   EnvelopeConfig.from_cfg(cfg), verbose=False)
    pool = pools['train']
    fl = FrequencySoftConstraintLoss.from_cfg(cfg)
    H, f0, valid = fl.H, fl.f0, fl.valid

    pk, y = collect_peaks(pool, args.episodes, args.n_shot)
    N, K = pk.shape
    print('=' * 78)
    print('诊断 L_freq：SNR=%.1f dB  n_shot=%d  episodes=%d  ->  N=%d  K=%d'
          % (args.snr, args.n_shot, args.episodes, N, K))
    print('类别分布 %s' % torch.bincount(y, minlength=4).tolist())
    print('H (Hn=%d, C=%d):\n%s' % (H.shape[0], H.shape[1], H.numpy()))
    print('f0 = %s' % f0.numpy().round(3).tolist())
    print('valid = %s  (False 的类是 NC，其惩罚被置零)' % valid.tolist())
    print('=' * 78)

    # ------------------------------------------------------------------
    # D. 物理先验本身的判别力（先看，因为它决定上限）
    # ------------------------------------------------------------------
    print('\n[D] 物理先验判别力（未掩码 rel，只统计 3 个故障类）')
    with torch.no_grad():
        _, parts = frequency_soft_constraint(
            torch.zeros(N, 4), pk, H, f0, valid=None, return_parts=True)
        rel_un = parts['rel']                                   # (N, C) 未掩码
    rel_true = rel_un.gather(1, y.view(-1, 1)).squeeze(1)
    rel_best_wrong = rel_un.scatter(1, y.view(-1, 1), 1e9).min(1).values
    margin = rel_best_wrong - rel_true                          # >0 表示先验支持正确类
    fault = y != 0
    sub = rel_un[:, 1:4]
    pred_fault = sub.argmin(1) + 1
    acc_fault = float((pred_fault[fault] == y[fault]).float().mean())
    print('  正确类 rel 均值      = %.4f' % float(rel_true.mean()))
    print('  最优错误类 rel 均值  = %.4f' % float(rel_best_wrong.mean()))
    print('  margin(=错-对) 均值  = %+.4f   中位数 %+.4f   margin>0 占比 %.1f%%'
          % (float(margin.mean()), float(margin.median()),
             100.0 * float((margin > 0).float().mean())))
    print('  **纯物理先验**在故障类上的 top-1 准确率 = %.4f   (n=%d)'
          % (acc_fault, int(fault.sum())))
    print('  解读：margin>0 占比 就是"物理先验站在正确类这边"的样本比例；')
    print('        它是 L_freq 能把 logits 推向正确类的**理论上限**。')

    # ------------------------------------------------------------------
    # A. 梯度量级对比（随饱和程度）
    # ------------------------------------------------------------------
    print('\n[A] ∂L/∂logits 范数对比（logits = gap·(2·onehot(y)-1)，gap 越大越饱和）')
    print('  %-7s %-12s %-13s %-13s %-11s' % ('gap', 'L_freq', '|g_freq|', '|g_ce|', 'g_freq/g_ce'))
    base = (F.one_hot(y, 4).float() * 2 - 1)                     # 正确类 +1，其余 -1
    for gap in (0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 10.0, 20.0, 40.0):
        z1 = (base * gap).clone().requires_grad_(True)
        lf = frequency_soft_constraint(z1, pk, H, f0, valid)
        gf = float(torch.autograd.grad(lf, z1)[0].norm())
        z2 = (base * gap).clone().requires_grad_(True)
        ce = F.cross_entropy(z2, y)
        gce = float(torch.autograd.grad(ce, z2)[0].norm())
        print('  %-7.1f %-12.6f %-13.3e %-13.3e %-11.3e'
              % (gap, float(lf), gf, gce, gf / (gce + 1e-30)))
    print('  解读：g_freq 在 gap 增大（模型变确信）时急剧衰减 —— 这是 softmax 加权')
    print('        的固有性质：一旦 p 饱和成 one-hot，∂p/∂z→0，约束就"说不上话"。')

    # ------------------------------------------------------------------
    # B. 只用 L_freq 优化 logits，能推到正确类吗
    # ------------------------------------------------------------------
    print('\n[B] 只优化 L_freq（无 CE），看 logits 最终落到哪一类')
    for lr in (0.05, 0.2):
        z = torch.zeros(N, 4, requires_grad=True)
        opt = torch.optim.Adam([z], lr=lr)
        for _ in range(300):
            opt.zero_grad()
            frequency_soft_constraint(z, pk, H, f0, valid).backward()
            opt.step()
        with torch.no_grad():
            pred = z.argmax(1)
            acc = float((pred == y).float().mean())
            lf_fin = float(frequency_soft_constraint(z, pk, H, f0, valid))
        print('  lr=%-5.2f  收敛后 L_freq=%.6f  与真实标签一致率=%.4f' % (lr, lf_fin, acc))
    print('  解读：一致率低 -> 单靠 L_freq 无法区分这几类（先验判别力不足）；')
    print('        一致率高 -> 先验有判别力，问题只出在梯度量级/权重 λ。')

    # ------------------------------------------------------------------
    # C. 先验敏感性：打乱 H 的列
    # ------------------------------------------------------------------
    print('\n[C] 控制变量：把 H 的列打乱（错误物理先验），L_freq 变化多少')
    perm = torch.tensor([0, 2, 1, 3])                            # 交换 IR007 / OR007
    H_sh = H[:, perm].clone()
    z0 = (base * 4.0).clone()
    lf_ok = float(frequency_soft_constraint(z0, pk, H, f0, valid))
    lf_sh = float(frequency_soft_constraint(z0, pk, H_sh, f0, valid))
    zr = torch.randn(N, 4)
    lf_ok_r = float(frequency_soft_constraint(zr, pk, H, f0, valid))
    lf_sh_r = float(frequency_soft_constraint(zr, pk, H_sh, f0, valid))
    print('  饱和 logits (gap=4):  真先验 %.6f   打乱先验 %.6f   相对变化 %+.2f%%'
          % (lf_ok, lf_sh, 100.0 * (lf_sh - lf_ok) / (abs(lf_ok) + 1e-30)))
    print('  随机 logits:          真先验 %.6f   打乱先验 %.6f   相对变化 %+.2f%%'
          % (lf_ok_r, lf_sh_r, 100.0 * (lf_sh_r - lf_ok_r) / (abs(lf_ok_r) + 1e-30)))
    print('  解读：饱和时打乱先验几乎不改变损失 -> 约束在模型确信后形同虚设。')

    # ------------------------------------------------------------------
    # 结论提示
    # ------------------------------------------------------------------
    print('\n' + '=' * 78)
    print('判读路线：')
    print('  · 若 [D] 的 acc_fault 高（例如 >0.9）但 [B] 一致率低 -> 先验有信息，')
    print('    是"梯度被 CE 淹没 / 权重 λ 太小 / warmup 太晚"的问题；')
    print('  · 若 [D] 的 acc_fault 本就低 -> 该先验在此噪声下判别力不足，')
    print('    需要换更强的物理量（如包络谱幅值比、谐波能量占比）而不是只看峰位。')
    print('=' * 78)


if __name__ == '__main__':
    main()
