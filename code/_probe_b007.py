# -*- coding: utf-8 -*-
"""诊断：CWRU 的 B007（滚动体故障）谱峰到底落在哪？

背景（实测事实）
----------------
干净数据（无噪声）下，用"理论特征频率谐波"做判据时：
    OR007 召回 = 1.0000   IR007 召回 = 0.9496   **B007 召回 = 0.0000**
说明 B007 用的物理量（H 第 3 列 = BSF 的 1/2/3 阶谐波，即 4.7135×fr 及其倍数）
**与数据对不上**。这不是噪声问题。

本脚本直接打印各类样本包络谱的 top 峰频率，并标注它们与候选理论频率的距离，
用来定位 B007 的真实特征频率落在哪。候选（@1797 rpm, fr=29.95 Hz）：
    fr        = 29.95   （转频）
    FTF       = 11.93   （保持架，0.3983×fr）
    BSF       = 70.58   （滚动体自转，2.3567×fr）
    2×BSF     = 141.17  （4.7135×fr，CWRU 官方给的就是这个）
    BPFO      = 107.36  （3.5848×fr）
    BPFI      = 162.19  （5.4152×fr）
以及 BSF 被 FTF 调制的边带：2×BSF ± k×FTF。
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                                          # noqa: E402
import yaml                                                                 # noqa: E402

from data.cwru_loader import prepare, NOMINAL_RPM                           # noqa: E402
from data.bearing_freq import fault_frequencies                             # noqa: E402
from data.envelope_spectrum import EnvelopeConfig, envelope_spectrum        # noqa: E402
from data.envelope_spectrum import detect_peaks                             # noqa: E402

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
    ap.add_argument('--per-class', type=int, default=8, dest='per_class')
    ap.add_argument('--topk', type=int, default=8)
    ap.add_argument('--snr', type=float, default=None, help='默认不加噪（干净数据）')
    args = ap.parse_args(argv)

    cfg = load_cfg()
    pools = prepare(cfg, force=False, verbose=False)
    if args.snr is not None:
        from data.cwru_loader import add_noise_to_pools
        pools = add_noise_to_pools(pools, args.snr, 12345,
                                   EnvelopeConfig.from_cfg(cfg), verbose=False)
    pool = pools['train']
    ecfg = EnvelopeConfig.from_cfg(cfg)
    rpm = NOMINAL_RPM[0]
    ff = fault_frequencies(rpm)
    print('rpm=%s  fr=%.2f Hz' % (rpm, ff['f_r']))

    cand = {
        'fr': ff['f_r'],
        'FTF': ff['FTF'],
        'BSF': ff['BSF'],
        '2BSF': ff['BSF_2x'],
        'BPFO': ff['BPFO'],
        'BPFI': ff['BPFI'],
    }
    print('候选理论频率: ' + '  '.join('%s=%.2f' % (k, v) for k, v in cand.items()))
    print()

    classes = ['NC', 'IR007', 'OR007', 'B007']
    rng = np.random.default_rng(0)
    for c in (1, 2, 3):                      # 只看 3 个故障类
        ci = np.flatnonzero(pool.y == c)
        pick = rng.choice(ci, size=min(args.per_class, ci.size), replace=False)
        print('=' * 78)
        print('类 %-6s (n=%d)  包络谱 top-%d 峰：' % (classes[c], len(pick), args.topk))
        near_hist = {k: 0 for k in cand}
        for i in pick:
            freqs, amp = envelope_spectrum(pool.X[i], ecfg)
            pf, pa, _ = detect_peaks(freqs, amp, ecfg, args.topk)
            tags = []
            for f in pf:
                # 找最近的候选频率（相对容差 3%）
                best_k, best_d = None, 1e9
                for k, v in cand.items():
                    d = abs(f - v) / v
                    if d < best_d:
                        best_k, best_d = k, d
                if best_d < 0.03:
                    near_hist[best_k] += 1
                    tags.append('%.1f(%s)' % (f, best_k))
                else:
                    tags.append('%.1f(?)' % f)
            print('  ' + '  '.join(tags))
        tot = sum(near_hist.values())
        print('  -> top 峰的最近候选命中统计: %s  (共 %d 个峰)'
              % ({k: v for k, v in near_hist.items() if v}, tot))

    print('\n' + '=' * 78)
    print('判读：')
    print('  · IR007 应命中 BPFI，OR007 应命中 BPFO —— 若这两类命中良好，说明流程没问题；')
    print('  · B007 若大量命中 2BSF 则理论没错、是容差/阶数问题；')
    print('    若命中 FTF / BSF / 2BSF±FTF，说明滚动体故障的特征频率取法需要改；')
    print('    若大量落空(?) 且无规律，说明该类的包络谱主体并不在特征频率上。')
    print('=' * 78)


if __name__ == '__main__':
    main()
