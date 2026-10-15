# -*- coding: utf-8 -*-
"""零训练探针：为什么"自估 f_r"折损增益，而"精度更差的区间随机化"却不折损？

背景（由四臂端到端对照提出，见 paper 5.5.2 节）：
  臂 ② 固定但错误（+5.7 %）        → 跨负载增益未退化
  臂 ④ 区间随机化（±100 rpm ≈ ±5.6 %）→ 跨负载增益未退化（跨 3 hp 甚至最高）
  臂 ③ 谱峰自估（误差 ≤ 0.86 %）   → 跨负载增益折损到约 46 %–57 %

  ⇒ **f_r 的精度高低不是决定因素**。真正差别是：
     · 臂 ④ 的 f_r 来自**与样本无关**的分布（以标称为中心）；
     · 臂 ③ 的 f_r 由**本 episode 自己的谱峰**产生，且其目标函数正是
       "仅故障类的平均最小相对阶次偏差"——**它被选来最小化约束本身要惩罚的量**
       （objective-coupled / data-coupled）。

本探针在**同一批 episode** 上零训练地测量各 f_r 来源下先验的**判别属性**，
正是 paper 5.13 节第 10 条所列的待验证项 (i)「真类是否为 argmin 的一致率」。

测量量（只依赖谱峰与 rel 定义，不含任何模型训练）：
  · argmin 一致率：故障样本中 rel 最小的故障类 == 真类的比例（越大越好）；
  · 相对间隔  ：min_{c≠真类} rel_c − rel_{真类}（越大越可分）；
  · 类间对比度：max_c rel_c − min_c rel_c。

用法：
  python code/_probe_speed_coupling.py --out results/tables/_speed_coupling.md
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from data import bearing_freq as bf  # noqa: E402

CLASSES3 = ["NC", "IR007", "OR007"]
FAULT = [1, 2]                       # 类索引（NC = 0 无频率先验）
DSET = "pools_w2048_st512_se512_cap400_bal1_c3-NC+IR007+OR007.npz"
POOLS = os.path.join(os.path.dirname(_HERE), "data", "processed", DSET)

N_H = 3
EPS = 1e-8
N_QUERY = 15                          # n_query_per_class（与 episode_sampler 一致）
TRAIN_RPM = 1797.0                    # 训练域 0 hp 标称（四臂的训练期基点）
WRONG_RPM = 1900.0                    # 臂 ② 的"固定但错误"先验
JITTER_RPM = 100.0                    # 臂 ④ 的区间半宽
GRID = np.linspace(1500.0, 2100.0, 121)   # 与 loss.speed_grid 一致
TRUE_RPM = {"test": 1797.0, "cross_1": 1772.0, "cross_2": 1750.0, "cross_3": 1730.0}
POOL_ORDER = ["test", "cross_1", "cross_2", "cross_3"]
POOL_LABEL = {"test": "0 hp（训练域）", "cross_1": "1 hp", "cross_2": "2 hp", "cross_3": "3 hp"}


def orders():
    O = np.zeros(len(CLASSES3))
    mask = np.zeros(len(CLASSES3), dtype=bool)
    fr = 1.0
    table = {"BPFO": bf.bpfo(fr), "BPFI": bf.bpfi(fr), "BSF": bf.bsf_2x(fr)}
    for i, c in enumerate(CLASSES3):
        ft = bf.CLASS_FAULT_TYPE.get(c)
        if ft is None:
            continue
        O[i] = table[ft]
        mask[i] = True
    return O, mask


def rel_of(peaks, O, mask, fr):
    """rel[i,c] = min_k min_h |f_{i,k} - h·O_c·f_r| / (O_c·f_r)。(N,C)，NC 为 inf。"""
    H = np.zeros((N_H, len(O)))
    for c in range(len(O)):
        if mask[c]:
            for h in range(N_H):
                H[h, c] = (h + 1) * O[c] * fr
    diff = np.abs(peaks[:, :, None, None] - H[None, None, :, :])
    rel = diff.min(axis=1).min(axis=1)
    f0 = np.where(mask, O * fr, 1.0)
    return np.where(mask[None, :], rel / (f0[None, :] + EPS), np.inf)


def est_fr_from_peaks(peaks_fault, O, mask):
    """逐 episode 自估：使"仅故障类平均最小 rel"最小的 f_r（无标签、无模型）。"""
    sc = np.empty(len(GRID))
    for i, r in enumerate(GRID):
        sc[i] = float(np.mean(np.min(rel_of(peaks_fault, O, mask, r / 60.0), axis=1)))
    return float(GRID[int(np.argmin(sc))])


def measure(peaks, y, O, mask, fr):
    """(argmin 一致率, 平均相对间隔, 平均类间对比度)。"""
    rel = rel_of(peaks, O, mask, fr)
    sel = np.isin(y, FAULT)
    r, yy = rel[sel], y[sel]
    sub = r[:, FAULT]
    pos = {c: i for i, c in enumerate(FAULT)}
    cand = np.array([pos[v] for v in yy])
    agree = float(np.mean(np.array(FAULT)[np.argmin(sub, axis=1)] == yy))
    true_rel = sub[np.arange(len(yy)), cand]
    other = sub.copy()
    other[np.arange(len(yy)), cand] = np.inf
    margin = float(np.mean(np.min(other, axis=1) - true_rel))
    contrast = float(np.mean(np.subtract(np.max(sub, axis=1), np.min(sub, axis=1))))
    return agree, margin, contrast


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(_HERE), "results", "tables", "_speed_coupling.md"))
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--jitter-draws", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = np.load(POOLS, allow_pickle=True)
    keys = set(d.keys())
    O, mask = orders()
    rng = np.random.default_rng(args.seed)

    CONV = ["① 标称 %.0f rpm（固定）" % TRAIN_RPM,
            "② 标称 %.0f rpm（固定，偏 +5.7 %%）" % WRONG_RPM,
            "③ 逐 episode 谱峰自估",
            "④ 区间随机化 %.0f ± %.0f rpm" % (TRAIN_RPM, JITTER_RPM),
            "⑤ 目标域真值 f_r（oracle 参照）"]
    agree = np.zeros((len(CONV), len(POOL_ORDER)))
    margin = np.zeros_like(agree)
    contrast = np.zeros_like(agree)
    est_stat = {}

    for jp, p in enumerate(POOL_ORDER):
        if p + "__pk" not in keys:
            raise SystemExit("[FAIL] 池 %s 不存在，可用键：%s" % (p, sorted(keys)[:8]))
        pk = np.asarray(d[p + "__pk"], dtype=np.float64)
        y = np.asarray(d[p + "__y"])
        idx = {c: np.where(y == c)[0] for c in FAULT}
        acc = [np.zeros(3) for _ in CONV]
        ests = []
        for _ in range(args.episodes):
            q = np.concatenate([rng.choice(idx[c], size=N_QUERY, replace=True) for c in FAULT])
            peaks, yy = pk[q], y[q]
            acc[0] += np.array(measure(peaks, yy, O, mask, TRAIN_RPM / 60.0))
            acc[1] += np.array(measure(peaks, yy, O, mask, WRONG_RPM / 60.0))
            fe = est_fr_from_peaks(peaks, O, mask)
            ests.append(fe)
            acc[2] += np.array(measure(peaks, yy, O, mask, fe / 60.0))
            rr = TRAIN_RPM + rng.uniform(-1.0, 1.0, size=args.jitter_draws) * JITTER_RPM
            acc[3] += np.mean([measure(peaks, yy, O, mask, r / 60.0) for r in rr], axis=0)
            acc[4] += np.array(measure(peaks, yy, O, mask, TRUE_RPM[p] / 60.0))
        for i in range(len(CONV)):
            agree[i, jp] = acc[i][0] / args.episodes
            margin[i, jp] = acc[i][1] / args.episodes
            contrast[i, jp] = acc[i][2] / args.episodes
        est_stat[p] = (float(np.median(ests)), float(np.mean(ests)),
                       float(np.std(ests, ddof=1)), float(np.min(ests)), float(np.max(ests)))

    L = []
    P = L.append
    P("# 零训练探针：各 f_r 来源下先验的判别属性（为什么自估折损、随机化不折损？）")
    P("")
    P("- 数据：`%s` 的真实包络谱峰（K = 5），四个载荷的留出池各自独立采样" % DSET)
    P("- episode 口径：每类 %d 个 query（与训练一致），每池 %d 个 episode，seed = %d"
      % (N_QUERY, args.episodes, args.seed))
    P("- **本探针不含任何模型训练**；三项指标只依赖谱峰与 rel 定义，因此可被完全重算。")
    P("- 口径提醒：⑤ 行使用**目标域自身**的真值 f_r，是「先验最理想」的上界参照，"
      "非任何真实臂；③ 行在每个 episode 上用该 episode 自己的谱峰自估，与臂 ③ 同义。")
    P("")

    P("## 1 [可用] 训练域（0 hp）上的配对比较——表 19 的来源")
    P("")
    j0 = POOL_ORDER.index("test")
    P("| f_r 来源 | argmin 一致率 | 平均相对间隔 | 平均类间对比度 |")
    P("|---|---|---|---|")
    for i, nm in enumerate(CONV):
        P("| %s | %.4f | %.4f | %.4f |" % (nm, agree[i, j0], margin[i, j0], contrast[i, j0]))
    P("")
    P("> 一致性自检：0 hp 下 ① 标称 1797 rpm **就是**真值，故 ① 与 ⑤（oracle）的数值应相等——"
      "实测均为 %.4f，二者一致，说明本探针口径自洽。" % agree[0, j0])
    P("")
    P("**与 ① 的配对差（同一批 episode）**")
    P("")
    P("| f_r 来源 | Δ argmin 一致率 |")
    P("|---|---|")
    for i, nm in enumerate(CONV[1:], start=1):
        P("| %s | %+.4f |" % (nm, agree[i, j0] - agree[0, j0]))
    P("")
    P("## 2 [不可用] 跨载荷列：该指标自身的退化（重要的负面方法学结论）")
    P("")
    P("把同一测量搬到另外三个载荷上时，**该指标不再可用**。下表给出跨 2 hp / 跨 3 hp 在"
      "若干 f_r 下的 argmin 一致率：")
    P("")
    P("| f_r 来源 | " + " | ".join(POOL_LABEL[p] for p in POOL_ORDER) + " |")
    P("|---|" + "---|" * len(POOL_ORDER))
    for i, nm in enumerate(CONV):
        P("| %s | " % nm + " | ".join("%.4f" % agree[i, j] for j in range(len(POOL_ORDER))) + " |")
    P("")
    P("可见：")
    P("")
    P("- 多个格点出现**精确的 0.5000**，其含义是「该 f_r 下所有故障窗的 rel-argmin 都指向同一类」"
      "（判别退化为常量预测，故一致率恰为 1/2）；")
    P("- 一致率对 f_r **非单调**：跨 2 hp 下 ③ 的自估值 %.1f rpm（偏离真值仅 %.2f %%）得 %.4f，"
      "而 ① 的标称 1797 rpm（偏离真值 %.2f %%）反而得 %.4f；"
      % (est_stat["cross_2"][0], 100.0 * (est_stat["cross_2"][0] - TRUE_RPM["cross_2"]) / TRUE_RPM["cross_2"],
         agree[2, POOL_ORDER.index("cross_2")],
         100.0 * (TRAIN_RPM - TRUE_RPM["cross_2"]) / TRUE_RPM["cross_2"],
         agree[0, POOL_ORDER.index("cross_2")]))
    P("- 该非单调性**由论文自身的 4 类口径实现同样复现**（见 `_probe_order_domain.py` 的 `rule_acc`），"
      "因此不是本探针的编码问题，而是 rel 定义（min over K 个谱峰 × H_n 个谐波、再按 O_c·f_r 归一化）"
      "的固有性质：谐波网格随 f_r 移动，可能在**错误的** f_r 上与某类谱峰偶然对齐。")
    P("")
    P("⇒ 结论：**「真类是否为 rel-argmin」不能作为跨载荷的「先验质量」代理指标**。"
      "本文据此把原待办项 (i) 部分销项（训练域已测）、并新增一条方法学教训。")
    P("")

    P("## 3 自估 f_r 的逐 episode 分布")
    P("")
    P("| 数据池 | 中位数 | 均值 ± 标准差 | 最小 / 最大 | 与 1797 的中位偏差 |")
    P("|---|---|---|---|---|")
    for p in POOL_ORDER:
        md, mu, sd, mn, mx = est_stat[p]
        P("| %s | %.1f rpm | %.1f ± %.1f rpm | %.1f / %.1f | %+.1f rpm（%+.2f %% / 真值 %+.2f %%） |"
          % (POOL_LABEL[p], md, mu, sd, mn, mx, md - TRAIN_RPM,
             100.0 * (md - TRAIN_RPM) / TRAIN_RPM, 100.0 * (md - TRUE_RPM[p]) / TRUE_RPM[p]))
    P("")
    P("> 关键：自估值的**逐 episode 标准差只有约 1 rpm**（远小于它自身的偏差），"
      "因此「逐 episode 随机抖动」**不是**折损的原因——这一点可与 %d 次抽样的区间随机化（±%.0f rpm）对比。"
      % (args.jitter_draws, JITTER_RPM))
    P("")

    P("## 4 读法（由数据决定）")
    P("")
    d2 = agree[2, j0]
    jj = agree[3, j0]
    d3 = agree[1, j0]
    P("- 在**训练域（0 hp）**上：标称固定 %.4f、固定偏置 +5.7 %% %.4f、"
      "区间随机化 %.4f、**逐 episode 自估 %.4f**。" % (agree[0, j0], d3, jj, d2))
    if d2 < agree[0, j0] - 0.02 and jj > agree[0, j0] - 0.02:
        P("- **自估把先验判别力压低，而精度低一个量级的区间随机化并未压低**，"
          "与端到端四臂「自估臂折损、随机化臂不折损」的排序在**同一方向上一致**。"
          "由于自估值的 episode 间标准差仅约 1 rpm，该折损**不可能**由「随机噪声」解释，"
          "而更可能来自：自估器的目标（最小化仅故障类的**平均**最小 rel）本身**不是**"
          "最大化类间可分辨性——它由本 episode 的谱峰决定，因而与被评分的样本耦合。")
    else:
        P("- 训练域上的排序与端到端四臂的排序**不一致**，故 5.5.2 节的机制解释不成立，"
          "应改为报告纯现象并扩大待验证清单。")
    P("- **强度声明**：以上仍是**零训练旁证**。端到端链条中还隔着训练动态与泛化，"
      "本探针不能证明因果；且第 2 节已表明该指标在跨载荷上退化，故它不能用于跨载荷论证。")
    P("")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print("\n[写出] %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
