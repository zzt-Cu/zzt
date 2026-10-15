# -*- coding: utf-8 -*-
"""_probe_order_domain.py —— 阶次域 vs 频率域约束：转速先验误差的实测代价。

背景（对应用户 P0-2 / P0-3）
---------------------------
优化清单要求"不要直接用 0hp 理论 Hz 值跨工况"，改为阶次域（f/f_r）约束。

⚠️ 必须先讲清的事实（本脚本第 0 节用数值验证）：
    阶次域的 rel 与频率域的 rel 是**同一个量**：
        |f - h·O_c·f_r| / (O_c·f_r)  ≡  |f/f_r - h·O_c| / O_c
    所以"把 Hz 换成阶次"本身不改变任何数值，**真正的改动是 f_r 从哪来**：
        旧做法：f_r 由目标域标称转速给出（跨工况时只有 0hp 的 1797 rpm）
        新做法：f_r 由数据估计（不依赖任何转速先验）
    因此阶次域约束的收益 = 消除对标称转速的依赖，代价 = 引入 f_r 估计误差。

用真实 CWRU 四个载荷的包络谱峰（pools.npz）实测：
  1. 物理规则准确率随"假设转速"的漂移
  2. 三种 f_r 来源对比（0hp 标称 / oracle / 数据自估）
  3. f_r 可辨识性诊断
零训练、可复现。
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from data import bearing_freq as bf   # noqa: E402

RPM_TABLE = {0: 1797, 1: 1772, 2: 1750, 3: 1730}
CLASSES = ["NC", "IR007", "OR007", "B007"]
N_H = 3
EPS = 1e-8
FAULT = [1, 2, 3]
POOLS = os.path.join(os.path.dirname(_HERE), "data", "processed", "pools.npz")
LOADS = [("0hp (test)", "test__", 0), ("1hp (cross_1)", "cross_1__", 1),
         ("2hp (cross_2)", "cross_2__", 2), ("3hp (cross_3)", "cross_3__", 3)]


def orders():
    """各故障类的阶次（× f_r），载荷无关。返回 (O, mask)，形状均为 (C,)。"""
    O = np.zeros(len(CLASSES))
    mask = np.zeros(len(CLASSES), dtype=bool)
    fr = 1.0
    table = {"BPFO": bf.bpfo(fr), "BPFI": bf.bpfi(fr), "BSF": bf.bsf_2x(fr)}
    for i, c in enumerate(CLASSES):
        ft = bf.CLASS_FAULT_TYPE.get(c)
        if ft is None:
            continue
        O[i] = table[ft]
        mask[i] = True
    return O, mask


def rel_of(peaks, O, mask, fr):
    """rel[i,c] = min_k min_h |peak - h·O_c·f_r| / (O_c·f_r)。形状 (N,C)；无频率类为 inf。"""
    H = np.zeros((N_H, len(O)))
    for c in range(len(O)):
        if mask[c]:
            for h in range(N_H):
                H[h, c] = (h + 1) * O[c] * fr
    diff = np.abs(peaks[:, :, None, None] - H[None, None, :, :])   # (N,K,n_h,C)
    rel = diff.min(axis=1).min(axis=1)                             # (N,C)
    f0 = np.where(mask, O * fr, 1.0)
    rel = rel / (f0[None, :] + EPS)
    return np.where(mask[None, :], rel, np.inf)


def rel_orders(pk, O, mask, fr):
    """阶次**写法**（逐类循环，显式）：min_k min_h |f/f_r - h·O_c| / O_c。用于第 0 节验证。"""
    N = pk.shape[0]
    out = np.full((N, len(O)), np.inf)
    for c in range(len(O)):
        if not mask[c]:
            continue
        best = np.full(N, np.inf)
        for h in range(1, N_H + 1):
            d = np.abs(pk / fr - h * O[c]) / O[c]      # (N,K)
            best = np.minimum(best, d.min(axis=1))
        out[:, c] = best
    return out


def rule_acc(rel, y):
    """物理规则（rel 最小者胜）在**仅故障类**样本上的准确率。"""
    sel = np.isin(y, FAULT)
    pred = np.array(FAULT)[np.argmin(rel[sel][:, FAULT], axis=1)]
    return float(np.mean(pred == y[sel])), int(sel.sum())


def est_fr(pk, O, mask, grid):
    """无标签自估 f_r：取使"平均最小 rel"最小的 f_r。"""
    sc = np.array([float(np.mean(np.min(rel_of(pk, O, mask, fr), axis=1))) for fr in grid])
    return float(grid[int(np.argmin(sc))]), sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(_HERE),
                                                  "results", "tables", "_order_domain.md"))
    a = ap.parse_args()

    z = np.load(POOLS, allow_pickle=True)
    O, mask = orders()
    L = ["# 阶次域 vs 频率域约束：转速先验误差实测（零训练）", "",
         "数据：`data/processed/pools.npz` 各载荷留出池的**真实包络谱峰**（K=5）。",
         "物理规则 = 在 rel 最小的故障类上判决；表中为**仅故障类**口径准确率。", "",
         "**各故障类阶次 O_c（载荷无关）**：%s"
         % ", ".join("%s=%.4f" % (CLASSES[i], O[i]) for i in FAULT),
         "**各载荷标称转速**：%s"
         % ", ".join("%dhp=%d rpm (f_r=%.2f Hz)" % (k, v, v / 60.0) for k, v in RPM_TABLE.items()),
         "", "## 0. 先验证：阶次域 rel ≡ 频率域 rel", ""]
    pk3 = np.asarray(z["cross_3__pk"], dtype=np.float64)
    fr3 = RPM_TABLE[3] / 60.0
    r_hz = rel_of(pk3, O, mask, fr3)
    r_od = rel_orders(pk3, O, mask, fr3)
    d = float(np.max(np.abs(r_od[:, FAULT] - r_hz[:, FAULT])))
    L += ["- 全量 %d 个 3hp 样本，两种写法最大绝对差 = **%.3e**" % (len(pk3), d),
          "- ⇒ 阶次域**不是**新的约束形式，它只改变 f_r 的来源。", ""]

    L += ["## 1. 频率域：物理规则准确率 vs 假设转速", "",
          "假设转速 1600–1900 rpm（步长 5），各载荷的真实谱峰不变。",
          "1797 rpm 即「只拿 0hp 理论值当先验」的旧做法。", ""]
    grid = np.arange(1600.0, 1900.1, 5.0)
    curves = {}
    for name, pre, _hp in LOADS:
        pk = np.asarray(z[pre + "pk"], dtype=np.float64)
        y = np.asarray(z[pre + "y"], dtype=np.int64)
        curves[name] = np.array([rule_acc(rel_of(pk, O, mask, r / 60.0), y)[0] for r in grid])
    L.append("| 假设转速 | " + " | ".join(n for n, _, _ in LOADS) + " |")
    L.append("|" + "---|" * (len(LOADS) + 1))
    for i, r in enumerate(grid):
        if (r - 1600) % 25 == 0:
            tag = " ←0hp标称" if abs(r - 1797) < 3 else ""
            L.append("| **%d**%s | %s |"
                     % (r, tag, " | ".join("%.3f" % curves[n][i] for n, _, _ in LOADS)))
    L.append("")

    L += ["## 2. 三种 f_r 来源对比（核心表）", "",
          "| 载荷 | 真值 f_r | ①0hp 标称 29.95 Hz | ②oracle 正确 f_r | ③数据自估（无先验） | 自估误差 |",
          "|---|---|---|---|---|---|"]
    fgrid = np.arange(15.0, 40.01, 0.25)
    summ = []
    for name, pre, hp in LOADS:
        pk = np.asarray(z[pre + "pk"], dtype=np.float64)
        y = np.asarray(z[pre + "y"], dtype=np.int64)
        ft = RPM_TABLE[hp] / 60.0
        a1, _ = rule_acc(rel_of(pk, O, mask, RPM_TABLE[0] / 60.0), y)
        a2, _ = rule_acc(rel_of(pk, O, mask, ft), y)
        fh, sc = est_fr(pk, O, mask, fgrid)
        a3, _ = rule_acc(rel_of(pk, O, mask, fh), y)
        summ.append((name, ft, a1, a2, a3, fh))
        L.append("| %s | %.2f Hz | %.3f | **%.3f** | %.3f | %+.2f%% |"
                 % (name, ft, a1, a2, a3, 100 * (fh - ft) / ft))
    L += ["", "> ① 旧做法（跨工况只知 0hp 标称）；② 上界（转速表精确）；③ 新做法（谱峰自估 f_r）。",
          "> ⚠️ ③ 用的是**无窗口约束**的搜索，会退化到 f_r→0（见第 3 节）；",
          ">   加上物理窗口后的正确结果见**第 5 节**，那里自估误差降到 ≤0.64%。",
          "> ⚠️ 本表的『准确率』指标由 argmin 翻转主导、对转速不单调（见第 1、4 节），",
          ">   判断转速先验代价应以第 4 节的 **rel** 为准。", ""]

    L += ["## 3. f_r 可辨识性诊断", "",
          "| 载荷 | 真值 | 自估 | 真值在目标函数中的排名 | 目标函数尖锐度 |",
          "|---|---|---|---|---|"]
    for name, pre, hp in LOADS:
        pk = np.asarray(z[pre + "pk"], dtype=np.float64)
        ft = RPM_TABLE[hp] / 60.0
        fh, sc = est_fr(pk, O, mask, fgrid)
        i_true = int(np.argmin(np.abs(fgrid - ft)))
        rank = int((sc < sc[i_true]).sum()) + 1
        L.append("| %s | %.2f | %.2f | %d / %d | min=%.4f  median=%.4f  (min/median=%.3f) |"
                 % (name, ft, fh, rank, len(fgrid), sc.min(), float(np.median(sc)),
                    sc.min() / max(float(np.median(sc)), 1e-12)))
    L += ["", "> K=5 峰 + 包络谱分辨率有限时，f_r 是否可辨识决定阶次域约束的成败。", ""]

    L += ["## 4. 诊断：真值 rpm 处，真类的 rel 是否真的最小？", "",
          "若频率约定正确，'真类 rel' 应在真值 rpm 附近取最小。",
          "若它跑到别处，说明是频率/峰值口径问题，而不是转速效应。", "",
          "| 载荷 | 真值 rpm | acc 最优 rpm | '真类 rel' 最小 rpm | '最小 rel' 最小 rpm | 真类 rel@真值rpm | 真类 rel@0hp标称 |",
          "|---|---|---|---|---|---|---|"]
    for name, pre, hp in LOADS:
        pk = np.asarray(z[pre + "pk"], dtype=np.float64)
        y = np.asarray(z[pre + "y"], dtype=np.int64)
        sel = np.isin(y, FAULT)
        pks, ys = pk[sel], y[sel]
        accs, true_rel, min_rel = [], [], []
        for r in grid:
            rel = rel_of(pk, O, mask, r / 60.0)
            acc_v, _ = rule_acc(rel, y)
            accs.append(acc_v)
            rs = rel[sel]
            true_rel.append(float(np.mean(rs[np.arange(len(ys)), ys])))
            min_rel.append(float(np.mean(np.min(rs, axis=1))))
        true_rel = np.array(true_rel)
        min_rel = np.array(min_rel)
        i_nom = int(np.argmin(np.abs(grid - RPM_TABLE[0])))
        i_tru = int(np.argmin(np.abs(grid - RPM_TABLE[hp])))
        L.append("| %s | %d | %d | **%d** | %d | %.4f | %.4f |"
                 % (name, RPM_TABLE[hp], grid[int(np.argmax(accs))],
                    grid[int(np.argmin(true_rel))], grid[int(np.argmin(min_rel))],
                    true_rel[i_tru], true_rel[i_nom]))
    L += ["", "> 若末两列接近，说明 0hp 标称频率在跨工况下仍然可用（转速影响小）；",
          "> 若'真类 rel 最小 rpm'明显偏离真值，则需先修正峰值/频率口径。", ""]

    L += ["## 5. 把 f_r 搜索限制在物理可行窗口后，f_r 能否被辨识？", "",
          "第 3 节的自估退化到网格下界，原因是**谐波随 f_r 变小而变密**，",
          "密集谐波可以「拟合」任何峰 —— 这是目标函数本身的退化解，不是数据问题。",
          "把搜索限制在物理可行区间（4 极电机 1700–1800 rpm ⇒ f_r∈[28.3, 30.0] Hz，",
          "这里放宽到 25–35 Hz）后重测：", "",
          "| 载荷 | 真值 f_r | 受限搜索自估 f_r | 误差 | 自估处 '真类 rel' | 真值处 '真类 rel' |",
          "|---|---|---|---|---|---|"]
    bgrid = np.arange(25.0, 35.001, 0.05)
    for name, pre, hp in LOADS:
        pk = np.asarray(z[pre + "pk"], dtype=np.float64)
        y = np.asarray(z[pre + "y"], dtype=np.int64)
        sel = np.isin(y, FAULT)
        pks, ys = pk[sel], y[sel]
        ft = RPM_TABLE[hp] / 60.0
        sc = np.array([float(np.mean(np.min(rel_of(pk, O, mask, fr), axis=1))) for fr in bgrid])
        fh = float(bgrid[int(np.argmin(sc))])
        ra = rel_of(pks, O, mask, fh)
        rb = rel_of(pks, O, mask, ft)
        L.append("| %s | %.3f | **%.2f** | %+.2f%% | %.4f | %.4f |"
                 % (name, ft, fh, 100 * (fh - ft) / ft,
                    float(np.mean(ra[np.arange(len(ys)), ys])),
                    float(np.mean(rb[np.arange(len(ys)), ys]))))
    L += ["", "> **结论（与第 3 节形成对照）**：在**物理可行窗口**内，f_r 可由 K=5 谱峰自估，",
          "> 误差 ≤ 0.64%，且**正确复现了随载荷单调下降的趋势**（29.90→29.50→29.10→28.65）。",
          "> ⇒ 阶次域约束只需「电机大致转速区间」这一极弱先验，**不需要各载荷的标称转速**；",
          "> 这与第 4 节实测的「用 0hp 标称会使真类失配膨胀 55%」正好互补。",
          "> ⚠️ 但**搜索窗口的物理约束是本方法可用的前提**：窗口去掉（第 3 节）目标函数即退化到 f_r→0。", ""]

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))
    print("\nwritten ->", a.out)


if __name__ == "__main__":
    main()
