# -*- coding: utf-8 -*-
"""从 results_reference/ 的逐种子结果表重新绘制论文的 3 张图（600 dpi）。

用法：
    python visualization/figures.py                 # 输出到 <repo>/results/figures/
    python visualization/figures.py --out DIR --dpi 600

图与论文的对应关系：
    fig1_main.png     图 1  主结果的跨工况柱状图（16 seeds，误差棒 = 95 % CI）
    fig2_randfreq.png 图 2  随机频率对照：逐种子配对差与组均值
    fig3_arms.png     图 3  转速先验来源四臂对照（Δ = CNN-Freq − CNN-CE）

说明：误差棒统一为「16 个种子 seed 级准确率均值的 95 % 置信区间」（配对 t 分布，df = n-1），
与论文图注一致，不是标准差、也不是标准误。
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["savefig.bbox"] = "tight"

DIRS = ["in_acc", "x1_acc", "x2_acc", "x3_acc"]
LABELS = ["同工况 0 hp", "跨 1 hp", "跨 2 hp", "跨 3 hp"]
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CM = 1 / 2.54


def read(fn: str):
    df = pd.read_csv(os.path.join(ROOT, "results_reference", fn), encoding="utf-8-sig")
    ce = df[df.method == "CNN-CE"].set_index("seed").sort_index()
    fq = df[df.method == "CNN-Freq"].set_index("seed").sort_index()
    s = np.array(sorted(ce.index.intersection(fq.index)))
    return s, ce, fq


def ci95(x: np.ndarray) -> float:
    """均值的 95 % 置信区间半宽（t 分布，df = n-1）。"""
    n = len(x)
    return stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)


# ---------------------------------------------------------------- 图 1
def fig_main(out_dir, dpi):
    s, ce, fq = read("runs_b_3c_s12_sh5_s8_s16.csv")
    x = np.arange(len(DIRS))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8.6 * CM, 4.6 * CM))
    for k, (mat, name, color) in enumerate([(ce, "CNN-CE（基线）", "#4C72B0"),
                                            (fq, "CNN-Freq（本文）", "#DD8452")]):
        m = np.array([mat[c].mean() for c in DIRS])
        e = np.array([ci95(mat[c].values) for c in DIRS])
        ax.bar(x + (k - 0.5) * w, m, w, yerr=e, capsize=2, label=name, color=color,
               edgecolor="black", linewidth=0.5, error_kw=dict(lw=0.8))
        for xi, mi in zip(x + (k - 0.5) * w, m):
            ax.text(xi, mi + 0.012, "%.3f" % mi, ha="center", va="bottom", fontsize=5.2)
    ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=6.5)
    ax.set_ylabel("准确率", fontsize=7); ax.set_ylim(0, 0.82)
    ax.tick_params(labelsize=6.5)
    ax.legend(fontsize=6, frameon=False, loc="upper right", ncol=2)
    ax.grid(axis="y", ls=":", lw=0.4, alpha=0.6); ax.set_axisbelow(True)
    p = os.path.join(out_dir, "fig1_main.png")
    fig.savefig(p, dpi=dpi); fig.savefig(p.replace(".png", ".pdf")); plt.close(fig)
    return p


# ---------------------------------------------------------------- 图 2
def fig_randfreq(out_dir, dpi):
    s1, ce1, fq1 = read("runs_b_3c_s12_sh5_s8_s16.csv")
    s2, ce2, fq2 = read("runs_b_3c_s12_sh5_jit_s16.csv")
    cs = np.array(sorted(set(s1.tolist()) & set(s2.tolist())))
    d_real = np.array([fq1.loc[cs, c].values - ce1.loc[cs, c].values for c in DIRS]) * 100
    d_rand = np.array([fq2.loc[cs, c].values - ce2.loc[cs, c].values for c in DIRS]) * 100
    x = np.arange(len(DIRS))
    fig, ax = plt.subplots(figsize=(8.6 * CM, 4.4 * CM))
    rng = np.random.default_rng(0)
    for k, (D, name, color) in enumerate([(d_real, "真实频率", "#C44E52"),
                                          (d_rand, "随机频率", "#55A868")]):
        for i in range(len(DIRS)):
            jit = (rng.random(len(cs)) - 0.5) * 0.22
            ax.scatter(np.full(len(cs), x[i] + (k - 0.5) * 0.34) + jit, D[i],
                       s=4.5, color=color, alpha=0.75, linewidths=0)
        ax.errorbar(x + (k - 0.5) * 0.34, D.mean(axis=1), yerr=[ci95(D[i]) for i in range(len(DIRS))],
                    fmt="D", ms=3.2, color=color, capsize=2, lw=0.9, label=name, zorder=5)
    ax.axhline(0, color="black", lw=0.7, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=6.5)
    ax.set_ylabel(r"逐种子配对差 $\Delta$ (pt)", fontsize=7)
    ax.tick_params(labelsize=6.5)
    ax.legend(fontsize=6, frameon=False, ncol=2, loc="lower left")
    ax.grid(axis="y", ls=":", lw=0.4, alpha=0.6); ax.set_axisbelow(True)
    p = os.path.join(out_dir, "fig2_randfreq.png")
    fig.savefig(p, dpi=dpi); fig.savefig(p.replace(".png", ".pdf")); plt.close(fig)
    return p


# ---------------------------------------------------------------- 图 3
def fig_arms(out_dir, dpi):
    arms = [("runs_b_3c_s12_sh5_s8_s16.csv", "① fixed 标称 1797", "#4C72B0"),
            ("runs_ord3c_s12_sh5_rpm1900_s16.csv", "② fixed 写错 1900", "#DD8452"),
            ("runs_ord3c_s12_sh5_est_s16.csv", "③ estimate 自估", "#55A868"),
            ("runs_ord3c_s12_sh5_jit100_s16.csv", "④ jitter ±100", "#C44E52")]
    x = np.arange(len(DIRS)); w = 0.19
    fig, ax = plt.subplots(figsize=(8.6 * CM, 4.6 * CM))
    for k, (fn, name, color) in enumerate(arms):
        s, ce, fq = read(fn)
        d = np.array([fq[c].values - ce[c].values for c in DIRS]) * 100
        m = d.mean(axis=1); e = np.array([ci95(d[i]) for i in range(len(DIRS))])
        ax.bar(x + (k - 1.5) * w, m, w, yerr=e, capsize=1.6, label=name, color=color,
               edgecolor="black", linewidth=0.5, error_kw=dict(lw=0.7))
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=6.5)
    ax.set_ylabel(r"$\Delta$ = CNN-Freq − CNN-CE (pt)", fontsize=6.6)
    ax.tick_params(labelsize=6.5)
    ax.legend(fontsize=5.6, frameon=False, ncol=2, loc="upper left")
    ax.grid(axis="y", ls=":", lw=0.4, alpha=0.6); ax.set_axisbelow(True)
    p = os.path.join(out_dir, "fig3_arms.png")
    fig.savefig(p, dpi=dpi); fig.savefig(p.replace(".png", ".pdf")); plt.close(fig)
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "figures"))
    ap.add_argument("--dpi", type=int, default=600)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for fn in (fig_main, fig_randfreq, fig_arms):
        p = fn(a.out, a.dpi)
        from PIL import Image
        W, H = Image.open(p).size
        print("%-46s %dx%d px  → 9 cm 宽时约 %.0f dpi" % (p, W, H, W / (9 / 2.54)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
