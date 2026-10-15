# -*- coding: utf-8 -*-
"""物理先验"生效却有害"的解析式机制诊断（零训练，纯 numpy）。

为什么不需要前向传播
--------------------
`L_freq` 的输入里只有 ``logits`` 依赖模型参数；``peak_f`` 是离线预计算并缓存的常数。
于是（``fault_only=True``，softmax 只在有理论频率的类上归一化）::

    L_freq(theta) = sum_{c in V} p_c(theta) * rel_c ,   rel_c 与 theta 无关
    =>  dL_freq / d logit_c = p_c * (rel_c - r_bar) ,   r_bar = sum_k p_k rel_k
        dL_freq / d logit_c = 0                      (c 不属于 V)

闭式梯度 ⇒ 只要拿到已保存的 ``collect_logits / collect_peak``，
就能**精确**算出每个测试样本在最优 epoch 处的物理项梯度，无需复现训练。

要回答的问题
------------
上一轮实测：把 λ 提前到 epoch 12 之后，物理项**首次真正进入被报告的模型**
（权重不再逐位相同），但 3 类精度反而下降 1.3pt。为什么"生效"等于"有害"？

把样本按两个正交维度切开：**模型对不对** × **物理规则对不对**，共 4 格。
分母里同时给出 CE 的 logit 梯度 ``p - onehot(y)``，于是可以逐格回答：

  * 该修的样本（模型错 & 物理对）上，物理项的梯度有多大？够不够翻盘？
  * 会不会顺手把已经对的样本（模型对 & 物理错）改坏？
  * 两个梯度方向的夹角 cos(g_ce, g_freq) —— **是不是在对着干？**

用法
----
python _probe_prior_grad.py --glob "bnd3c_fastlam_snrm8_*,bnd4c_fastlam_snrm8_*" \
                            --out results/tables/_prior_grad.md
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _probe_boundary as pb                                    # noqa: E402

LARGE = 1.0e6


# ----------------------------------------------------------------------
# 表格化的物理频率（含 NC 占位列）
# ----------------------------------------------------------------------
def build_tables(bf, rpm: float, classes: Sequence[str], n_harmonics: int = 3,
                 use_bsf_2x: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 ``(H, f0, valid)``。

    ``H``: ``(n_harmonics, C)``，``H[h, c] = (h+1) * f_theory(c)``；NC 列为 NaN。
    ``f0``: ``(C,)`` 基频；NC 为 NaN。``valid``: ``(C,)`` bool。
    """
    Hd, f0d = pb._theory(bf, rpm, list(classes), n_harmonics, use_bsf_2x)
    C = len(classes)
    H = np.full((n_harmonics, C), np.nan)
    f0 = np.full(C, np.nan)
    valid = np.zeros(C, dtype=bool)
    for i, c in enumerate(classes):
        if c in Hd:
            H[:, i] = Hd[c][:n_harmonics]
            f0[i] = f0d[c]
            valid[i] = True
    return H, f0, valid


def rel_matrix(peak: np.ndarray, H: np.ndarray, f0: np.ndarray,
               valid: np.ndarray) -> np.ndarray:
    """与损失**完全同口径**的相对偏差 ``(M, C)``。

    ``rel[m, c] = min_k min_h |pk_k - H[h,c]| / f0[c]``；空缺峰（<=0）取 LARGE。
    NC 列（无理论频率）置 inf —— 它在 ``fault_only`` 口径下不参与 softmax。
    """
    M, K = peak.shape
    pk = np.where(peak > 0, peak, LARGE).astype(np.float64)          # (M,K)
    n_h, C = H.shape
    diff = np.abs(pk[:, :, None, None] - H[None, None, :, :])        # (M,K,Hn,C)
    rel = (diff / (f0[None, None, None, :] + 1e-8)).min(axis=2).min(axis=1)  # (M,C)
    rel[:, ~valid] = np.inf
    return rel


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ----------------------------------------------------------------------
# 逐 run 分析
# ----------------------------------------------------------------------
def analyze_run(run_dir: str, bf, rpm: float) -> Optional[Dict[str, object]]:
    d = pb._load_run(run_dir)
    if d is None:
        return None
    logits = np.asarray(d['logits'], dtype=np.float64)               # (M,C)
    y = np.asarray(d['y'], dtype=np.int64)
    peak = d['peak']
    if peak is None:
        print('!! %s 缺 collect_peak，跳过' % d['run_id'])
        return None
    peak = np.asarray(peak, dtype=np.float64)
    n = int(logits.shape[1])
    classes = pb.MASTER_CLASSES[:n]
    H, f0, valid = build_tables(bf, rpm, classes)
    if int(valid.sum()) < 2:
        return None
    rel = rel_matrix(peak, H, f0, valid)                             # (M,C)

    vi = np.where(valid)[0]                                          # 故障类下标
    vcls = [classes[i] for i in vi]

    # ---- 只保留"每个样本的真类都在故障类里"的部分（NC 无理论频率）----
    m = np.isin(y, vi)
    if int(m.sum()) == 0:
        return None
    yv = y[m]
    lgv = logits[np.ix_(m, vi)]                                      # (Mv,Cv)
    relv = rel[np.ix_(m, vi)]                                        # (Mv,Cv)

    p = softmax(lgv)                                                 # fault_only 口径
    rbar = (p * relv).sum(axis=1, keepdims=True)
    g_freq = p * (relv - rbar)                                       # (Mv,Cv) 闭式梯度
    # CE 的 logit 梯度：L_CE 在**全类** softmax 上算，故梯度维度为 C
    p_full = softmax(logits[m])
    oh = np.zeros_like(p_full)
    oh[np.arange(yv.size), yv] = 1.0
    g_ce = p_full - oh                                              # (Mv,C)

    pred_m = lgv.argmax(axis=1)                                      # 限定故障类的模型判决
    pred_p = relv.argmin(axis=1)                                     # 物理峰位规则
    y_local = np.asarray([list(vi).index(int(t)) for t in yv], dtype=np.int64)
    ok_m = pred_m == y_local
    ok_p = pred_p == y_local

    nf, nce = np.linalg.norm(g_freq, axis=1), np.linalg.norm(g_ce, axis=1)
    # 夹角：g_freq 补零到全类维度后与 g_ce 比较
    gf_full = np.zeros_like(g_ce)
    gf_full[:, vi] = g_freq
    num = (gf_full * g_ce).sum(axis=1)
    den = nf * nce + 1e-30
    cos = num / den

    cells = {}
    for nm, mask in (('M+ P+', ok_m & ok_p), ('M- P+', ~ok_m & ok_p),
                     ('M+ P-', ok_m & ~ok_p), ('M- P-', ~ok_m & ~ok_p)):
        k = int(mask.sum())
        if k == 0:
            cells[nm] = {'n': 0}
            continue
        cells[nm] = {
            'n': k,
            'g_freq_med': float(np.median(nf[mask])),
            'g_ce_med': float(np.median(nce[mask])),
            'ratio_med': float(np.median(nf[mask] / (nce[mask] + 1e-30))),
            'cos_mean': float(np.mean(cos[mask])),
            'margin_med': float(np.median((np.sort(lgv[mask], axis=1)[:, -1]
                                           - np.sort(lgv[mask], axis=1)[:, -2]))),
            # 该格里"物理项的损失代价差"：rel(物理选) - rel(模型选)
            'cost_gap_med': float(np.median(relv[mask, pred_p[mask]]
                                            - relv[mask, pred_m[mask]])),
        }

    return {
        'run_id': d['run_id'], 'n_class': n,
        'method': pb._parse_run_id(str(d['run_id']))[0],
        'seed': pb._parse_run_id(str(d['run_id']))[1],
        'tag': pb._parse_run_id(str(d['run_id']))[2],
        'classes': classes, 'fault_classes': vcls, 'n_fault_samples': int(m.sum()),
        'acc_model': float(ok_m.mean()), 'acc_phys': float(ok_p.mean()),
        'g_freq_med': float(np.median(nf)), 'g_ce_med': float(np.median(nce)),
        'ratio_med': float(np.median(nf / (nce + 1e-30))),
        'cos_mean': float(np.mean(cos)), 'cos_med': float(np.median(cos)),
        'cells': cells,
        # 若把模型判决整体替换成物理判决，净变化（=修好数 − 弄坏数）
        'net_swap': float((int((~ok_m & ok_p).sum()) - int((ok_m & ~ok_p).sum()))
                          / max(1, ok_m.size)),
        'rel_true_med': {classes[i]: float(np.median(relv[y_local == j, j]))
                         for j, i in enumerate(vi)},
    }


# ----------------------------------------------------------------------
# 报告
# ----------------------------------------------------------------------
def _f(v, p: int = 3) -> str:
    if v is None:
        return '-'
    try:
        f = float(v)
    except (TypeError, ValueError):
        return '-'
    return '%.*f' % (p, f) if f == f else '-'


def _e(v, p: int = 2) -> str:
    if v is None:
        return '-'
    f = float(v)
    return '%.*e' % (p, f) if f == f else '-'


def report(rows: List[Dict[str, object]], rpm: float) -> str:
    L: List[str] = []
    L.append('# 物理先验"生效却有害"的解析式机制诊断（零训练）\n')
    L.append('口径：`L_freq` 中 softmax 只在**有理论频率的类**上归一化（`fault_only=True`），'
             '与训练时一致；`rel` 与损失逐字同口径（最小峰 / 最小谐波 / 按基频归一）。')
    L.append('梯度为闭式解 `dL_freq/dlogit_c = p_c (rel_c - rbar)`，'
             '因此这里算的是**最优 epoch 处真实的反传信号**，不是近似。')
    L.append('统计仅覆盖真类 ∈ 故障类（NC 无理论频率）的 query 样本；转速 %g rpm。\n' % rpm)
    L.append('格子的含义：`M+` = 模型判决正确，`M-` = 错误；`P+` / `P-` = 物理规则对/错。')
    L.append('`净替换` = (物理能救回的样本数 − 物理会弄坏的样本数) / 样本数。\n')

    for r in rows:
        L.append('## %s（%d 类，%s seed=%s，%d 个故障类 query 样本）'
                 % (r['run_id'], r['n_class'], r['method'], r['seed'], r['n_fault_samples']))
        L.append('')
        L.append('| 指标 | 值 |')
        L.append('|---|---|')
        L.append('| 模型判决准确率（限定故障类 argmax） | %s |' % _f(r['acc_model']))
        L.append('| 物理峰位规则准确率（同批样本、同噪声） | %s |' % _f(r['acc_phys']))
        L.append('| **先验 − 模型** | %s |' % ('%+.3f' % (float(r['acc_phys']) - float(r['acc_model']))))
        L.append('| 净替换（整体换成物理判决） | %s |' % ('%+.3f' % r['net_swap']))
        L.append('| cos(g_CE, g_freq) 均值 | %s |' % _f(r['cos_mean']))
        L.append('| ‖g_freq‖ / ‖g_CE‖ 中位数 | %s |' % _e(r['ratio_med']))
        L.append('')
        L.append('| 格 | n | 占比 | ‖g_freq‖ 中位 | ‖g_CE‖ 中位 | 幅度比中位 | cos 均值 '
                 '| logit margin 中位 | 物理项代价差中位 |')
        L.append('|---|---|---|---|---|---|---|---|---|')
        tot = max(1, int(r['n_fault_samples']))
        for nm in ('M- P+', 'M- P-', 'M+ P-', 'M+ P+'):
            c = r['cells'][nm]                                    # type: ignore[index]
            if not c or c.get('n', 0) == 0:
                L.append('| **%s** | 0 | 0.0%% | - | - | - | - | - | - |' % nm)
                continue
            L.append('| **%s** | %d | %.1f%% | %s | %s | %s | %s | %s | %s |'
                     % (nm, c['n'], 100.0 * c['n'] / tot,
                        _e(c['g_freq_med']), _e(c['g_ce_med']), _e(c['ratio_med']),
                        _f(c['cos_mean']), _f(c['margin_med']), _f(c['cost_gap_med'])))
        L.append('')
        relm = ', '.join('%s=%.3f' % (k, v) for k, v in dict(r['rel_true_med']).items())  # type: ignore[arg-type]
        L.append('各类真样本的 `rel` 中位数：%s' % relm)
        L.append('')
        L.append('读法：')
        L.append('- **M- P+**（该修的样本）：若 `幅度比中位` 远小于 1，说明物理项在最该出手的地方'
                 '信号最弱 —— 梯度被 `p_c -> 0` 压掉了；')
        L.append('- **M+ P-**（不该动的样本）：若 `cos 均值` 为负，说明物理项在这里与 CE **对着干**，'
                 '每修一个样本要拿几个正确样本去换；')
        L.append('- `净替换 <= 0` ⇒ 把判决完全交给先验都不赚，优化器朝它走只会更差。')
        L.append('')

    # 汇总
    L.append('## 汇总（跨 run 均值）')
    L.append('')
    L.append('| run | 模型 acc | 物理 acc | 先验−模型 | 净替换 | cos 均值 | 幅度比中位 '
             '| M- P+ 幅度比 | M+ P- cos |')
    L.append('|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        c1 = r['cells']['M- P+']                                  # type: ignore[index]
        c2 = r['cells']['M+ P-']                                  # type: ignore[index]
        L.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s |'
                 % (r['run_id'], _f(r['acc_model']), _f(r['acc_phys']),
                    '%+.3f' % (float(r['acc_phys']) - float(r['acc_model'])),
                    '%+.3f' % r['net_swap'], _f(r['cos_mean']), _e(r['ratio_med']),
                    _e(c1.get('ratio_med')) if c1 and c1.get('n') else '-',
                    _f(c2.get('cos_mean')) if c2 and c2.get('n') else '-'))
    L.append('')
    return '\n'.join(L)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser('物理先验梯度机制诊断（零训练）')
    ap.add_argument('--logs', type=str, default=None)
    ap.add_argument('--glob', type=str, default='bnd3c_fastlam_snrm8_*,bnd4c_fastlam_snrm8_*')
    ap.add_argument('--out', type=str, default=None)
    ap.add_argument('--rpm', type=float, default=pb.RPM_IN)
    args = ap.parse_args(argv)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logs = args.logs or os.path.join(root, 'results', 'logs')
    bf = pb._load_bearing_freq()

    dirs: List[str] = []
    for pat in [x.strip() for x in args.glob.split(',') if x.strip()]:
        dirs.extend(pb._find_runs(logs, pat))
    seen = set()
    uniq = [d for d in dirs if not (d in seen or seen.add(d))]
    print('命中 %d 个 run' % len(uniq))

    rows: List[Dict[str, object]] = []
    for d in uniq:
        try:
            r = analyze_run(d, bf, args.rpm)
        except Exception as e:
            print('!! %s 分析失败：%s' % (os.path.basename(d), e))
            continue
        if r is not None:
            rows.append(r)
    if not rows:
        print('!! 没有可分析的 run')
        return 2
    text = report(rows, args.rpm)
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
