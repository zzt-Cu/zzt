# -*- coding: utf-8 -*-
"""离线验证：统一的「物理证据对数似然」能否修掉 NC 退化解、并提高判别力。

为什么改
--------
现行 L_freq = Σ_c p_c · rel_c（rel 为峰位相对偏差），且 NC 列被 valid 掩码**置 0**。
后果（已实测）：只优化 L_freq 时一致率恰为 0.25 —— 优化器发现"把所有样本判成 NC"
可以令损失为 0，于是坍缩。这是**设计缺陷**，不是调参问题。

新设计
------
对每个窗口，用包络谱算两个量：
    base  = 带内幅值中位数（噪声底）
    A_c   = Σ_h max(amp 落在 h·f0_c ±tol 内)      c ∈ {IR007, OR007, B007}
再构造**量纲统一**的物理对数似然（单位都是"相对噪声底的对数比"）：
    logL_c  = log(A_c / base)                     故障类：谐波处越强越像
    logL_NC = b_nc - log(A_max / base)            最强证据越弱，越像 NC
    q       = softmax([logL_NC, logL_IR, logL_OR, logL_BSF] / T)
L_freq = CE(q, p)（或 KL(q||p)）。

关键点：NC 有了**有限且可比较**的代价，"全判 NC"不再是全局最优。
b_nc 是唯一的标定参数，物理含义明确（"倾向 NC 的先验强度"）。

本脚本只做离线可行性验证（不训练）：
  1. 现行峰位法 vs 新幅值法的故障类 3-way 准确率；
  2. 4 类准确率随 b_nc 的扫描（给出最优值与平台区间）；
  3. 各类召回率；
  4. 干净数据 与 -8 dB 两档对照。

用法：
    python _probe_newloss.py
    python _probe_newloss.py --tol 0.03 --per-class 139
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                                          # noqa: E402
import yaml                                                                 # noqa: E402

from data.cwru_loader import prepare, add_noise_to_pools                    # noqa: E402
from data.envelope_spectrum import EnvelopeConfig, envelope_spectrum        # noqa: E402
from losses.physics_loss import FrequencySoftConstraintLoss                 # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CLASSES = ['NC', 'IR007', 'OR007', 'B007']


def load_cfg() -> dict:
    with open(os.path.join(HERE, 'configs', 'default.yaml'), encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg['freq']['shaft_rpm'] = {int(k): float(v) for k, v in cfg['freq']['shaft_rpm'].items()}
    for k in ('raw_dir', 'processed_dir'):
        cfg['data'][k] = os.path.join(ROOT, cfg['data'][k])
    for k, v in list(cfg['paths'].items()):
        cfg['paths'][k] = v if os.path.isabs(v) else os.path.join(ROOT, v)
    return cfg


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def run_one_tag(pool, sel, ecfg, H, f0, args, tag):
    X, y = pool.X[sel], pool.y[sel]
    N = len(y)
    Hn = H.shape[0]
    band = ecfg.band

    A = np.zeros((N, 3), dtype=np.float64)
    BASE = np.zeros(N, dtype=np.float64)
    for i in range(N):
        freqs, amp = envelope_spectrum(X[i], ecfg)
        m_band = (freqs >= band[0]) & (freqs <= band[1])
        BASE[i] = float(np.median(amp[m_band]))
        for k, c in enumerate((1, 2, 3)):
            a = 0.0
            for h in range(Hn):
                fh = float(H[h, c])
                if fh <= 0:
                    continue
                m = ((freqs >= fh * (1 - args.tol)) & (freqs <= fh * (1 + args.tol)))
                if m.any():
                    a += float(amp[m].max())
            A[i, k] = a

    # ---- 现行：峰位法 ----
    PF = pool.peak_f[sel]
    rel = np.full((N, 3), 1e9, dtype=np.float64)
    for i in range(N):
        pf = PF[i][PF[i] > 0]
        if pf.size == 0:
            continue
        for k, c in enumerate((1, 2, 3)):
            d = np.abs(pf[:, None] - H[:, c][None, :]) / (f0[c] + 1e-8)
            rel[i, k] = float(d.min())

    fault = y > 0
    pred_rel = rel.argmin(1) + 1
    acc_rel3 = float((pred_rel[fault] == y[fault]).mean())

    # ---- 新：幅值证据对数似然 ----
    logL_fault = np.log(A + 1e-9) - np.log(BASE + 1e-9)[:, None]          # (N,3)
    logL_nc_raw = -np.log(A.max(1) / BASE + 1e-9)                          # (N,)

    pred_amp3 = logL_fault.argmax(1) + 1
    acc_amp3 = float((pred_amp3[fault] == y[fault]).mean())

    print('\n' + '-' * 78)
    print('[%s]  样本 N=%d（每类 %d）    tol=±%.0f%%' % (tag, N, args.per_class, 100 * args.tol))
    print('  故障类 3-way 准确率:  峰位法(现行) %.4f   |   幅值证据法(新) %.4f'
          % (acc_rel3, acc_amp3))

    # 扫描 b_nc
    grid = np.arange(-4.0, 8.01, 0.1)
    best = (0.0, 0.0)
    rows = []
    for b in grid:
        lg = np.concatenate([(logL_nc_raw + b)[:, None], logL_fault], axis=1) / args.temp
        q = softmax(lg)
        pred = q.argmax(1)
        acc = float((pred == y).mean())
        rows.append((b, acc))
        if acc > best[1]:
            best = (float(b), acc)
    # 平台区间：最优附近 (±0.3 内精度损失 < 1e-3)
    plate = [b for b, a in rows if a >= best[1] - 1e-3]
    print('  4 类最优 b_nc = %+.1f  -> 准确率 %.4f   (平台区间 [%+.1f, %+.1f])'
          % (best[0], best[1], min(plate), max(plate)))

    b_star = best[0]
    lg = np.concatenate([(logL_nc_raw + b_star)[:, None], logL_fault], axis=1) / args.temp
    q = softmax(lg)
    pred = q.argmax(1)
    print('  各类召回@b_nc*:      ' + '  '.join(
        '%s=%.3f' % (CLASSES[c], float((pred[y == c] == c).mean())) for c in range(4)))
    # 现行设计对照：NC 列 rel 被 valid 掩码置 0 -> 对所有样本 argmin 都取到 NC(列 0)
    pred_cur = np.zeros(N, dtype=int)
    print('  对照·现行设计 4 类准确率 = %.4f（NC 列 rel 置 0 -> argmin 恒为 NC，退化解）'
          % float((pred_cur == y).mean()))

    # 只判故障类的退化风险检查：把 NC 强制纳入后，预测为 NC 的比例
    print('  预测为 NC 的比例 = %.3f（真实 NC 占 %.3f）'
          % (float((pred == 0).mean()), float((y == 0).mean())))
    return dict(tag=tag, acc_rel3=acc_rel3, acc_amp3=acc_amp3,
                b_nc=best[0], acc4=best[1])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-class', type=int, default=139, dest='per_class')
    ap.add_argument('--tol', type=float, default=0.02)
    ap.add_argument('--temp', type=float, default=1.0)
    args = ap.parse_args(argv)

    cfg = load_cfg()
    base_pools = prepare(cfg, force=False, verbose=False)
    ecfg = EnvelopeConfig.from_cfg(cfg)
    fl = FrequencySoftConstraintLoss.from_cfg(cfg)
    H = fl.H.numpy()
    f0 = fl.f0.numpy()

    rng = np.random.default_rng(0)
    sel = np.concatenate([
        rng.choice(np.flatnonzero(base_pools['train'].y == c),
                   size=min(args.per_class, int((base_pools['train'].y == c).sum())),
                   replace=False) for c in range(4)])

    print('=' * 78)
    print('新损失设计离线验证   （每类 %d 窗口，共 %d）' % (args.per_class, len(sel)))
    print('H:\n%s' % H.round(2))
    print('=' * 78)

    out = []
    out.append(run_one_tag(base_pools['train'], sel, ecfg, H, f0, args, '干净数据'))
    pooled = add_noise_to_pools(base_pools, -8.0, 12345, ecfg, verbose=False)
    out.append(run_one_tag(pooled['train'], sel, ecfg, H, f0, args, '-8 dB'))

    print('\n' + '=' * 78)
    print('汇总')
    print('  %-10s %-14s %-14s %-10s %-10s' % ('数据', '峰位法3way', '幅值法3way', '最优b_nc', '4类准确率'))
    for r in out:
        print('  %-10s %-14.4f %-14.4f %+-10.1f %-10.4f'
              % (r['tag'], r['acc_rel3'], r['acc_amp3'], r['b_nc'], r['acc4']))
    print('\n判读：若"幅值法3way"明显高于"峰位法3way"，且 4 类准确率 > 0.85，')
    print('      则新设计可行 -> 值得替换 physics_loss 并重建窗口池。')
    print('      若 4 类仍低，说明幅值证据在该 SNR 下不足以支撑 4 类，需再想别的物理量。')
    print('=' * 78)


if __name__ == '__main__':
    main()
