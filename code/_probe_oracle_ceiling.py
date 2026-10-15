# -*- coding: utf-8 -*-
"""_probe_oracle_ceiling.py —— 零训练测量「物理规则的信息上限」。

背景（论文 5.6）：论文用推理主张软惩罚式约束存在结构性上限——L_freq 的最小值点
就是物理规则分类器本身，故"不可能通过与 CNN 互补获得大幅增益"。

本脚本把该主张从**推理**变成**测量**：把冻结的 CNN-CE logits 与物理规则 one-hot
按权重 w 线性混合，扫描 w ∈ [0,1]，取准确率最大值。
  · 若最优 w 下增益 ≈ 0  ⇒ 规则确实没有可用信息，论文的"上限"论断成立；
  · 若最优 w 下增益显著 > 0 ⇒ 信息存在且可被最简单的后处理榨出，
    则"结构性互补不可能"与 5.13 的"不是优化失败"都需要修正，
    真正的瓶颈是**训练期的梯度通道**而非规则的信息量。

⚠️ 口径说明（写论文时必须声明）：
  · w 是在测试集上选的 ⇒ 结果是该混合族在测试集上的**上界**；
  · 只用冻结 logits 做线性混合 ⇒ 是信息量的**下界**（更复杂的利用还能更高）；
  · 含 NC 的完整口径下 w ≥ 0.5 会因"物理规则从不预测 NC"而使 argmax 恒为物理规则，
    故有效混合区间只有 w < 0.5。

用法：
    python _probe_oracle_ceiling.py                       # 默认跑主结果 + 四类档位
    python _probe_oracle_ceiling.py --out results/tables/_oracle_ceiling.md
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from data import bearing_freq as bf   # noqa: E402

RPM, N_H, EPS = 1797.0, 3, 1e-8

CASES = [
    ("b_3c_s12_sh5_s8", ["NC", "IR007", "OR007"], range(8),
     "主结果档位：−12 dB / 5-shot / 三类"),
    ("b_4c_s12_sh5", ["NC", "IR007", "OR007", "B007"], range(8),
     "四类档位：−12 dB / 5-shot / 四类（含先验无据的 B007）"),
]


def rel_matrix(peaks, classes):
    """rel[i, c] = min_k min_h |peak − H[h,c]| / f0[c]；NC 列置 inf（与损失口径一致）。"""
    H, mask = bf.theoretical_frequency_matrix(RPM, classes, N_H)
    Hm = np.asarray(H, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    diff = np.abs(peaks[:, :, None, None] - Hm[None, None, :, :])
    rel = diff.min(axis=1).min(axis=1) / (Hm[0][None, :] + EPS)
    rel[:, ~m] = np.inf
    return rel, m


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def sweep(tag, classes, seeds, fault_only, log_dir):
    """返回 (ws, acc_curve, n, 基线acc, 纯规则acc)。"""
    C = len(classes)
    V = np.arange(1, C)
    ws = np.round(np.arange(0, 1.0001, 0.02), 4)
    acc = np.zeros_like(ws)
    n_tot = 0
    for s in seeds:
        p = os.path.join(log_dir, "%s_CNN-CE_seed%d" % (tag, s), "arrays.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=True)
        lg, y, pk = z["collect_logits"], z["collect_y"], z["collect_peak"]
        if fault_only:
            sel = np.isin(y, V)
            lg, y, pk = lg[sel], y[sel], pk[sel]
        rel, _ = rel_matrix(pk, classes)
        p_ce = softmax(lg[:, V]) if fault_only else softmax(lg)
        e = np.zeros_like(p_ce)
        if fault_only:
            e[np.arange(len(y)), np.argmin(rel[:, V], axis=1)] = 1.0
        else:
            e[np.arange(len(y)), V[np.argmin(rel[:, V], axis=1)]] = 1.0
        for i, w in enumerate(ws):
            idx = np.argmax((1 - w) * p_ce + w * e, axis=1)
            # fault_only 口径下 p_ce 只有 |V| 列，argmax 返回**位置**下标，
            # 必须经 V 映射回原始类别下标才能与 y 比较（否则准确率会低于随机水平）。
            pred = V[idx] if fault_only else idx
            acc[i] += float(np.sum(pred == y))
        n_tot += len(y)
    acc /= max(1, n_tot)
    return ws, acc, n_tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default=os.path.join(os.path.dirname(_HERE),
                                                      "results", "logs"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(_HERE),
                                                 "results", "tables",
                                                 "_oracle_ceiling.md"))
    a = ap.parse_args()

    lines = ["# 物理规则的信息上限（零训练测量）", "",
             "把冻结的 CNN-CE logits 与物理规则 one-hot 线性混合，扫描权重 w。",
             "`Δ*` = 该混合族在测试集上的最优增益（w 在测试集上选取 ⇒ 是上界）。", ""]
    for tag, classes, seeds, title in CASES:
        for fault_only in (True, False):
            ws, acc, n = sweep(tag, classes, seeds, fault_only, a.log_dir)
            if n == 0:
                continue
            base, star = acc[0], int(np.argmax(acc))
            scope = "仅故障类（与表 11 同口径）" if fault_only else "含 NC 的完整口径"
            # 健全性检查：纯物理规则（w=1）若低于随机水平，说明类别下标映射错了
            # （fault_only 下 argmax 返回的是 V 内的位置下标，必须映射回原始下标）。
            chance = 1.0 / (len(classes) - 1 if fault_only else len(classes))
            if acc[-1] < chance - 1e-6:
                raise AssertionError(
                    "%s / %s：纯物理规则 acc=%.4f 低于随机水平 %.4f，"
                    "疑似下标映射错误" % (title, scope, acc[-1], chance))
            lines.append("## %s —— %s" % (title, scope))
            lines.append("")
            lines.append("- n = %d，w* = %.2f" % (n, ws[star]))
            lines.append("- 随机水平（参考）: %.4f" % chance)
            lines.append("- w=0（纯 CNN-CE）: %.4f" % base)
            lines.append("- w=1（纯物理规则）: %.4f" % acc[-1])
            lines.append("- **最优混合: %.4f  →  Δ* = %+.2f pt**" % (acc[star], 100 * (acc[star] - base)))
            lines.append("")
            lines.append("| w | acc | Δ(pt) |")
            lines.append("|---|---|---|")
            for i, w in enumerate(ws):
                if abs(w * 20 - round(w * 20)) < 1e-9:
                    lines.append("| %.2f | %.4f | %+.2f |" % (w, acc[i], 100 * (acc[i] - base)))
            lines.append("")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print("\nwritten ->", a.out)


if __name__ == "__main__":
    main()
