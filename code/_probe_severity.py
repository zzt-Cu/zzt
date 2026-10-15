# -*- coding: utf-8 -*-
"""验证：跨缺陷尺寸（0.007" -> 0.014" -> 0.021"）下，物理量是否仍然有效？

理由
----
BPFO/BPFI/BSF 只取决于**轴承几何与转速**，与缺陷尺寸无关；尺寸改变的是特征频率处的
**幅值**。所以"幅值型物理量"理论上应能跨尺寸迁移 —— 这是物理约束最可能体现价值的
方向（比"4 类分类精度"更有说服力，因为这里是一个真正的 domain shift）。

数据（0hp / 1797rpm，OR 统一取 @6:00 位置以保证可比）
------------------------------------------------------
    IR007=105   OR007@6=130   B007=118
    IR014=169   OR014@6=197   B014=185      (OR014 仅 @6:00 有数据)
    IR021=209   OR021@6=234   B021=222      (注意 0hp 是 234 不是 235)

做法
----
直接读 .mat -> segment(2048/512) -> z-score -> 用 @1797rpm 的理论频率算幅值证据
-> 在 3 个故障类之间取 argmax，看是否判对。**不改动项目代码**（纯离线验证）。

关键指标：3 个尺寸下各自的 3-way 准确率，以及"跨尺寸是否稳定"。
若 0.021" 与 0.007" 的准确率接近，说明该物理量对缺陷尺寸不敏感 —— 这正是
"物理约束提供 domain-invariant 先验"的直接证据。
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                                          # noqa: E402
import yaml                                                                 # noqa: E402

from data.cwru_loader import NOMINAL_RPM, cap_windows, load_de_signal       # noqa: E402
from data.cwru_loader import segment, zscore_windows                        # noqa: E402
from data.envelope_spectrum import EnvelopeConfig, envelope_spectrum        # noqa: E402
from losses.physics_loss import FrequencySoftConstraintLoss                 # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

FILES = {
    ('IR', '007'): 105, ('OR', '007'): 130, ('B', '007'): 118,
    ('IR', '014'): 169, ('OR', '014'): 197, ('B', '014'): 185,
    ('IR', '021'): 209, ('OR', '021'): 234, ('B', '021'): 222,
}
CLASS_COL = {'IR': 1, 'OR': 2, 'B': 3}        # H 的列索引（0 是 NC 占位）
SIZES = ['007', '014', '021']
TYPES = ['IR', 'OR', 'B']


def load_cfg() -> dict:
    with open(os.path.join(HERE, 'configs', 'default.yaml'), encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg['freq']['shaft_rpm'] = {int(k): float(v) for k, v in cfg['freq']['shaft_rpm'].items()}
    for k in ('raw_dir', 'processed_dir'):
        cfg['data'][k] = os.path.join(ROOT, cfg['data'][k])
    for k, v in list(cfg['paths'].items()):
        cfg['paths'][k] = v if os.path.isabs(v) else os.path.join(ROOT, v)
    return cfg


def evidence_matrix(X, ecfg, H, tol):
    """返回 (n, 3) 幅值证据，列序对应 H 的第 1/2/3 列（IR/OR/B）。"""
    Hn = H.shape[0]
    out = np.zeros((X.shape[0], 3), dtype=np.float64)
    for i in range(X.shape[0]):
        freqs, amp = envelope_spectrum(X[i], ecfg)
        for k, c in enumerate((1, 2, 3)):
            a = 0.0
            for h in range(Hn):
                fh = float(H[h, c])
                if fh <= 0:
                    continue
                m = (freqs >= fh * (1 - tol)) & (freqs <= fh * (1 + tol))
                if m.any():
                    a += float(amp[m].max())
            out[i, k] = a
    return out


def windows_of(code, raw, window, stride, cap, limit=None):
    sig, rec = load_de_signal(code, raw)
    W = segment(sig, window, stride)
    if limit is not None:
        W = W[:limit]
    W, _ = cap_windows(W, cap)
    return zscore_windows(W), rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-file', type=int, default=150, dest='per_file')
    ap.add_argument('--tol', type=float, default=0.02)
    ap.add_argument('--window', type=int, default=2048)
    ap.add_argument('--stride', type=int, default=512)
    args = ap.parse_args(argv)

    cfg = load_cfg()
    raw = cfg['data']['raw_dir']
    ecfg = EnvelopeConfig.from_cfg(cfg)
    fl = FrequencySoftConstraintLoss.from_cfg(cfg)
    H = fl.H.numpy()
    rpm = NOMINAL_RPM[0]
    print('=' * 78)
    print('跨缺陷尺寸物理量验证  rpm=%.0f  窗长=%d stride=%d  tol=±%.0f%%'
          % (rpm, args.window, args.stride, 100 * args.tol))
    print('H(@%.0frpm): %s' % (rpm, H[0].round(2).tolist()))
    print('=' * 78)

    missing = []
    for size in SIZES:
        for t in TYPES:
            code = FILES[(t, size)]
            if not os.path.exists(os.path.join(raw, '%d.mat' % code)):
                missing.append((t, size, code))
    if missing:
        print('⚠️  缺少文件，请先跑 _download_cwru_extra.py ：%s' % missing)
        return 1

    for snr in (None, -8.0):
        tag = '干净数据' if snr is None else ('SNR=%.0f dB' % snr)
        print('\n' + '=' * 78)
        print('【%s】' % tag)
        print('  %-6s %-8s %-10s %-10s %-10s %-10s' %
              ('尺寸', 'n', 'IR召回', 'OR召回', 'B召回', '3way准确率'))
        conf_all = np.zeros((3, 3), dtype=int)
        for size in SIZES:
            Xs, ys = [], []
            for k, t in enumerate(TYPES):
                X, _ = windows_of(FILES[(t, size)], raw, args.window, args.stride,
                                  args.per_file)
                if snr is not None:
                    rng = np.random.default_rng(12345)
                    p_s = (X ** 2).mean()
                    noise = rng.standard_normal(X.shape) * np.sqrt(
                        p_s / (10.0 ** (snr / 10.0)))
                    X = (X + noise).astype(np.float32)
                Xs.append(X)
                ys.append(np.full(X.shape[0], k))
            X = np.concatenate(Xs, 0)
            y = np.concatenate(ys, 0)
            E = evidence_matrix(X, ecfg, H, args.tol)
            pred = E.argmax(1)
            acc = float((pred == y).mean())
            recs = [float((pred[y == k] == k).mean()) for k in range(3)]
            for k in range(3):
                for j in range(3):
                    conf_all[k, j] += int(((y == k) & (pred == j)).sum())
            print('  %-6s %-8d %-10.3f %-10.3f %-10.3f %-10.3f'
                  % (size + '"', X.shape[0], recs[0], recs[1], recs[2], acc))
        print('  汇总混淆（行=真实 IR/OR/B，列=预测）：')
        for k, t in enumerate(TYPES):
            row = conf_all[k]
            tot = row.sum()
            if tot:
                print('    %-3s -> %s   (n=%d)' %
                      (t, '  '.join('%s=%.2f' % (TYPES[j], row[j] / tot) for j in range(3)), tot))

    print('\n' + '=' * 78)
    print('判读：')
    print('  · 若三个尺寸的 3way 准确率彼此接近（跨尺寸稳定），说明该物理量对缺陷尺寸')
    print('    不敏感 —— 这正是"物理约束提供 domain-invariant 先验"的直接证据；')
    print('  · 若 0.014"/0.021" 明显低于 0.007"，说明尺寸变了、物理判据也漂移；')
    print('  · 若 B 类在所有尺寸下都低，则它确实是该数据集上的硬短板（须如实报告）。')
    print('=' * 78)
    return 0


if __name__ == '__main__':
    sys.exit(main())
