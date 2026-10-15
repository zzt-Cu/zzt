# -*- coding: utf-8 -*-
"""验证：包络谱在特征频率处的**幅值证据**能否取代"峰位匹配"作为更强的物理先验。

背景（实测事实）
----------------
`_probe_freq_grad.py` 显示：-8 dB 下"峰位匹配"先验的故障类 top-1 只有 **0.6902**，
margin>0 占比 51.8%（≈抛硬币）。而文献里（arXiv:2508.07536 等）用的是特征频率处的
**幅值**（包络谱在 BPFO/BPFI 处有多强），而不是峰位。本脚本直接对比两者判别力，
用来决定"是否值得把 L_freq 从峰位匹配改成幅值证据"。

对比项
------
1. 故障类 top-1：幅值证据法 vs 峰位法；
2. NC 可分性：故障样本与 NC 样本的"最大归一化证据 maxE"分布是否分离
   —— 若可分，则 4 类都能覆盖，从根上解决"NC 列 rel 置零导致全判 NC"的缺陷；
3. 阈值法 4 类准确率。

用法
----
    python _probe_evidence.py --snr=-8 --per-class 200
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


def load_cfg() -> dict:
    with open(os.path.join(HERE, 'configs', 'default.yaml'), encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg['freq']['shaft_rpm'] = {int(k): float(v) for k, v in cfg['freq']['shaft_rpm'].items()}
    for k in ('raw_dir', 'processed_dir'):
        cfg['data'][k] = os.path.join(ROOT, cfg['data'][k])
    for k, v in list(cfg['paths'].items()):
        cfg['paths'][k] = v if os.path.isabs(v) else os.path.join(ROOT, v)
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--snr', type=float, default=-8.0)
    ap.add_argument('--per-class', type=int, default=200, dest='per_class')
    ap.add_argument('--tol', type=float, default=0.02, help='谐波邻域相对容差')
    args = ap.parse_args(argv)

    cfg = load_cfg()
    pools = prepare(cfg, force=False, verbose=False)
    if args.snr is not None:
        pools = add_noise_to_pools(pools, args.snr, 12345,
                                   EnvelopeConfig.from_cfg(cfg), verbose=False)
    pool = pools['train']
    ecfg = EnvelopeConfig.from_cfg(cfg)
    fl = FrequencySoftConstraintLoss.from_cfg(cfg)
    H = fl.H.numpy()
    f0 = fl.f0.numpy()
    valid = fl.valid.numpy().astype(bool)
    band = ecfg.band
    classes = ['NC', 'IR007', 'OR007', 'B007']

    rng = np.random.default_rng(0)
    sel = np.concatenate([
        rng.choice(np.flatnonzero(pool.y == c),
                   size=min(args.per_class, int((pool.y == c).sum())), replace=False)
        for c in range(4)])
    X, y, PF = pool.X[sel], pool.y[sel], pool.peak_f[sel]
    N = X.shape[0]

    print('=' * 78)
    print('物理量对比：SNR=%.1f dB  每类 %d 窗口  共 %d' % (args.snr, args.per_class, N))
    print('H (Hn=%d, C=%d)  谐波邻域容差 ±%.0f%%' % (H.shape[0], H.shape[1], 100 * args.tol))
    print('=' * 78)

    # ---------------- 幅值证据 E ----------------
    E = np.zeros((N, 4), dtype=np.float64)
    for i in range(N):
        freqs, amp = envelope_spectrum(X[i], ecfg)
        m_band = (freqs >= band[0]) & (freqs <= band[1])
        base = float(np.median(amp[m_band]))
        for c in range(4):
            if not valid[c]:
                continue
            tot = 0.0
            for h in range(H.shape[0]):
                fh = float(H[h, c])
                if fh <= 0:
                    continue
                m = (freqs >= fh * (1 - args.tol)) & (freqs <= fh * (1 + args.tol))
                if m.any():
                    tot += float(amp[m].max())
            E[i, c] = tot / (base + 1e-30)

    # ---------------- 峰位法 rel ----------------
    REL = np.full((N, 4), 1e9, dtype=np.float64)
    for i in range(N):
        pf = PF[i][PF[i] > 0]
        if pf.size == 0:
            continue
        for c in range(4):
            if not valid[c]:
                continue
            d = np.abs(pf[:, None] - H[:, c][None, :]) / (f0[c] + 1e-8)
            REL[i, c] = float(d.min())

    fault = y != 0
    nf = int(fault.sum())

    # ---- 1. 故障类 top-1 ----
    acc_amp = float((E[fault][:, 1:4].argmax(1) + 1 == y[fault]).mean())
    acc_rel = float((REL[fault][:, 1:4].argmin(1) + 1 == y[fault]).mean())
    print('\n[1] 故障类 3-way top-1 (n=%d)' % nf)
    print('    幅值证据法 : %.4f' % acc_amp)
    print('    峰位匹配法 : %.4f   <- _probe_freq_grad.py 里报的就是这一支' % acc_rel)
    print('    提升       : %+.4f (%+.1f%%)'
          % (acc_amp - acc_rel, 100 * (acc_amp - acc_rel) / (acc_rel + 1e-30)))

    # ---- 2. NC 可分性 ----
    maxE = E[:, 1:4].max(1)
    print('\n[2] 最大归一化证据 maxE 的分布（NC 判据的物理基础）')
    print('    故障样本: median=%.3f  p05=%.3f  p25=%.3f' %
          (np.median(maxE[fault]), np.percentile(maxE[fault], 5), np.percentile(maxE[fault], 25)))
    print('    NC  样本: median=%.3f  p75=%.3f  p95=%.3f' %
          (np.median(maxE[~fault]), np.percentile(maxE[~fault], 75), np.percentile(maxE[~fault], 95)))
    # 简单的分离度指标：用最优阈值下的 balanced accuracy
    ths = np.linspace(0, float(np.percentile(maxE[fault], 99)), 400)
    best_bal, best_th = 0.0, 0.0
    for t in ths:
        pos = maxE >= t                       # 判为故障
        tpr = float(pos[fault].mean())
        tnr = float((~pos[~fault]).mean())
        bal = 0.5 * (tpr + tnr)
        if bal > best_bal:
            best_bal, best_th = bal, float(t)
    print('    NC/故障 二分类最优阈值 tau=%.3f -> balanced acc = %.4f' % (best_th, best_bal))

    # ---- 3. 阈值法 4 类 ----
    best_acc, best_t = 0.0, 0.0
    for t in ths:
        pred = np.where(maxE < t, 0, E[:, 1:4].argmax(1) + 1)
        a = float((pred == y).mean())
        if a > best_acc:
            best_acc, best_t = a, float(t)
    print('\n[3] 阈值法 4 类准确率 = %.4f  (tau=%.3f)'
          % (best_acc, best_t))
    print('    对照·全判多数类 = %.4f' % float((y == np.bincount(y).argmax()).mean()))
    for c in range(4):
        m = y == c
        pred = np.where(maxE[m] < best_t, 0, E[m][:, 1:4].argmax(1) + 1)
        print('    每类召回 %-6s = %.4f  (n=%d)' % (classes[c], float((pred == c).mean()), int(m.sum())))

    print('\n' + '=' * 78)
    print('判读：')
    print('  · 若 [1] 幅值法 >> 峰位法（例如 >0.85），换物理量有效，值得改损失设计；')
    print('  · 若 [3] 的 4 类准确率明显高于 0.69，且 [2] 的 balanced acc 高，')
    print('    说明幅值证据天然覆盖 NC，可从根上消除"全判 NC"的退化解。')
    print('=' * 78)


if __name__ == '__main__':
    main()
