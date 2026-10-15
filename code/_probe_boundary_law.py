# -*- coding: utf-8 -*-
"""边界律：把"物理约束的增益 ΔAcc"回归到"先验相对模型的判别力优势"上。

动机
----
前面各阶段分散跑了很多 tag，每个 tag 单独看 Δ(Freq−CE) 都在 ±1.5pt 内来回跳，
单看任何一个都得不出结论。但如果把它们**放在同一根轴上排序**，
谱系应当显现出来。这根轴不是"难度"（SNR），而是**直接测出来的**

    prior_adv = acc_phys − acc_model

即"包络谱峰位规则"相对 CNN 的判别力优势（零训练、解析式，见 `_probe_prior_grad.py`）。

预期（可证伪）
--------------
  * `prior_adv > 0`（模型弱于先验）⇒ Δ > 0，尤其在**跨工况**列；
  * `prior_adv <= 0`（模型强于先验）⇒ Δ <= 0；
  * `prior_adv ≈ 0` ⇒ Δ 的方差 >> 均值，符号在种子间翻转。

两个正交的"模型强度"操纵子（用来验证这条律不是数据巧合）：
  * n_shot ∈ {1,5} —— query 谱峰不变，只有模型变强 ⇒ 干净的操纵；
  * SNR ∈ {0,-8,-12} dB —— 模型与先验同时退化。

用法
----
python _probe_boundary_law.py --out results/tables/_boundary_law.md
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _probe_boundary as pb                                    # noqa: E402
import _probe_prior_grad as pg                                  # noqa: E402


# ----------------------------------------------------------------------
def load_runs_csv(path: str) -> Dict[Tuple[str, int], Dict[str, float]]:
    """``{(method, seed): row}``。"""
    out: Dict[Tuple[str, int], Dict[str, float]] = {}
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            m = r.get('method')
            try:
                s = int(r.get('seed', -1))
            except (TypeError, ValueError):
                continue
            row: Dict[str, float] = {}
            for k, v in r.items():
                try:
                    row[k] = float(v)
                except (TypeError, ValueError):
                    row[k] = v                          # type: ignore[assignment]
            out[(str(m), s)] = row
    return out


def pair_deltas(tbl: Dict[Tuple[str, int], Dict[str, float]],
                metrics: Sequence[str]) -> Optional[Dict[str, object]]:
    """同 seed 配对求 CNN-Freq − CNN-CE。"""
    seeds = sorted({s for (m, s) in tbl if m == 'CNN-CE'}
                   & {s for (m, s) in tbl if m == 'CNN-Freq'})
    if not seeds:
        return None
    res: Dict[str, object] = {'seeds': seeds}
    for k in metrics:
        ce = [float(tbl[('CNN-CE', s)].get(k, float('nan'))) for s in seeds]
        fr = [float(tbl[('CNN-Freq', s)].get(k, float('nan'))) for s in seeds]
        d = np.asarray(fr, dtype=float) - np.asarray(ce, dtype=float)
        if not np.all(np.isfinite(d)):
            res[k] = float('nan')
            res[k + '_d'] = []
            continue
        res[k] = float(np.mean(fr) - np.mean(ce))
        res[k + '_ce'] = float(np.mean(ce))
        res[k + '_fr'] = float(np.mean(fr))
        res[k + '_d'] = d.tolist()
    return res


def _f(v, p: int = 4) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return '-'
    return '%.*f' % (p, f) if f == f else '-'


def _sgn(d: Sequence[float]) -> str:
    """正/负/零 计数，用来一眼看出符号是否翻转。"""
    a = [float(x) for x in d]
    if not a:
        return '-'
    return '%d+ / %d- / %d0' % (sum(1 for x in a if x > 1e-9),
                                sum(1 for x in a if x < -1e-9),
                                sum(1 for x in a if abs(x) <= 1e-9))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser('边界律：ΔAcc ~ 先验优势')
    ap.add_argument('--tags', type=str, default='all')
    ap.add_argument('--out', type=str, default=None)
    ap.add_argument('--rpm', type=float, default=pb.RPM_IN)
    args = ap.parse_args(argv)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tabdir = os.path.join(root, 'results', 'tables')
    logs = os.path.join(root, 'results', 'logs')

    files = sorted(glob.glob(os.path.join(tabdir, 'runs_*.csv')))
    # 过滤掉冒烟/临时产物：否则 `_tmp_jit`、`_tmp_probe4c` 这类 3-epoch 短跑会被当成
    # 正式 tag 混进总表，把"Δ≈0"的假数据当证据（本机实测踩过）。
    def _is_junk(n: str) -> bool:
        return n.startswith('_') or n == 'smoke' or 'diag' in n
    entries: List[Dict[str, object]] = []
    for fp in files:
        name = os.path.basename(fp)[len('runs_'):-len('.csv')]
        if _is_junk(name):
            continue
        if args.tags != 'all' and name not in [x.strip() for x in args.tags.split(',')]:
            continue
        tbl = load_runs_csv(fp)
        d = pair_deltas(tbl, ['in_acc', 'in_f1', 'x1_acc', 'x2_acc', 'x3_acc',
                              'in_win_acc', 'in_freq_consistency'])
        if d is None:
            continue
        d['tag'] = name
        # 先验优势（零训练解析式，逐 seed 平均）
        ad, dn = [], []
        dirs = pb._find_runs(logs, name + '_*')
        bf = pb._load_bearing_freq()
        for rd in dirs:
            if pb._parse_run_id(os.path.basename(rd))[0] != 'CNN-CE':
                continue
            rr = pg.analyze_run(rd, bf, args.rpm)
            if rr is None:
                continue
            ad.append(float(rr['acc_phys']) - float(rr['acc_model']))
            dn.append(float(rr['net_swap']))
        d['prior_adv'] = float(np.mean(ad)) if ad else float('nan')
        d['prior_adv_d'] = ad
        d['net_swap'] = float(np.mean(dn)) if dn else float('nan')
        d['n_run'] = len(dirs)
        entries.append(d)
    if not entries:
        print('!! 没有任何可配对的 tag')
        return 2

    L: List[str] = []
    L.append('# 边界律：物理约束增益 ΔAcc 与"先验相对模型的判别力优势"的关系\n')
    L.append('全部为**同 seed 配对**差值（CNN-Freq − CNN-CE），同一写死的 λ 调度'
             '（`warmup=4, step=2, max=0.5`，λ 在 epoch 索引 12 到 0.5）。')
    L.append('`先验优势` = 包络谱峰位规则准确率 − CNN-CE 准确率'
             '（零训练解析式，仅统计故障类 query）；`净替换` = 物理能救回 − 物理会弄坏。')
    L.append('⚠️ 先验优势 > 0 才意味着"信息在数据里而模型没用上"，才有增益空间。\n')

    L.append('## 1. 按先验优势排序的总表（核心）')
    L.append('')
    L.append('| tag | 先验优势 | 净替换 | Δ in-domain | 符号 | Δ x1 | Δ x2 | Δ x3 |')
    L.append('|---|---|---|---|---|---|---|---|')
    for d in sorted(entries, key=lambda x: -(x['prior_adv'] if x['prior_adv'] == x['prior_adv'] else -9)):
        L.append('| %s | %s | %s | %s | %s | %s | %s | %s |'
                 % (d['tag'], _f(d['prior_adv']), _f(d['net_swap']), _f(d['in_acc'], 5),
                    _sgn(d['in_acc_d']),              # type: ignore[arg-type]
                    _f(d['x1_acc'], 5), _f(d['x2_acc'], 5), _f(d['x3_acc'], 5)))
    L.append('')

    L.append('## 2. 逐种子配对明细（看符号是否翻转）')
    L.append('')
    L.append('| tag | seed | Δ in-domain | Δ x1 | Δ x2 | Δ x3 | 先验优势(该 seed) |')
    L.append('|---|---|---|---|---|---|---|')
    for d in entries:
        for i, s in enumerate(d['seeds']):        # type: ignore[arg-type]
            pa = (d['prior_adv_d'][i] if i < len(d['prior_adv_d']) else float('nan'))  # type: ignore[index]
            L.append('| %s | %d | %s | %s | %s | %s | %s |'
                     % (d['tag'], s,
                        _f(d['in_acc_d'][i], 5), _f(d['x1_acc_d'][i], 5),   # type: ignore[index]
                        _f(d['x2_acc_d'][i], 5), _f(d['x3_acc_d'][i], 5),   # type: ignore[index]
                        _f(pa)))
    L.append('')

    # ---- 相关性：把 (tag,seed) 作为独立观测 ----
    xs, ys_in, ys_cross, ys_all = [], [], [], []
    for d in entries:
        for i in range(len(d['prior_adv_d'])):    # type: ignore[arg-type]
            pa = d['prior_adv_d'][i]              # type: ignore[index]
            try:
                di = d['in_acc_d'][i]             # type: ignore[index]
                dx = d['x2_acc_d'][i]             # type: ignore[index]
            except (IndexError, KeyError):
                continue
            if pa != pa or di != di or dx != dx:
                continue
            xs.append(pa)
            ys_in.append(di)
            ys_cross.append(dx)
            ys_all.append(0.5 * di + 0.5 * dx)
    L.append('## 3. 相关性（观测 = (tag, seed)，n=%d）' % len(xs))
    L.append('')

    def _corr(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
        if len(a) < 3:
            return float('nan'), float('nan')
        aa, bb = np.asarray(a, float), np.asarray(b, float)
        if aa.std() == 0 or bb.std() == 0:
            return float('nan'), float('nan')
        r = float(np.corrcoef(aa, bb)[0, 1])
        # Pearson r 的 t 检验（n-2 自由度）
        n = len(aa)
        t = r * np.sqrt((n - 2) / max(1e-12, 1 - r * r))
        from math import lgamma, log, pi
        # 双尾 p（用 Student-t 的数值积分近似，避免依赖 scipy）
        def _tcdf(x: float, df: int) -> float:
            # 用连分式不现实，改用数值积分
            from math import gamma
            def integ(u: float) -> float:
                return (1.0 + u * u / df) ** (-(df + 1) / 2.0)
            N = 4000
            hi = abs(x)
            s = 0.0
            for k in range(N):
                u = hi * (k + 0.5) / N
                s += integ(u)
            s *= hi / N
            c = gamma((df + 1) / 2.0) / (np.sqrt(df * pi) * gamma(df / 2.0))
            p_one = 0.5 - c * s if x > 0 else 0.5 + c * s
            return p_one
        p = 2.0 * _tcdf(t, n - 2)
        return r, float(min(1.0, max(0.0, p)))

    for lbl, ys in (('Δ in-domain', ys_in), ('Δ cross 2hp', ys_cross), ('Δ 平均(in+cross2)', ys_all)):
        r, p = _corr(xs, ys)
        L.append('- **%s** vs 先验优势： Pearson r = %s，p ≈ %s' % (lbl, _f(r, 3), _f(p, 3)))
    L.append('')
    L.append('> 判读：r 显著为正 ⇒ "先验优势"确实能预测约束增益，边界律成立；')
    L.append('> r ≈ 0 ⇒ 增益与先验优势无关，前面的正负跳动只是种子噪声。')
    L.append('')

    text = '\n'.join(L)
    if args.out:
        out = os.path.normpath(args.out if os.path.isabs(args.out)
                               else os.path.join(root, args.out))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w', encoding='utf-8') as f:
            f.write(text + '\n')
        print('报告已写入：%s' % out)
    else:
        print(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
