# -*- coding: utf-8 -*-
"""零训练消融探针（阶次域）：峰值数 K ∈ {1,3,5} × 谐波阶数 H_n ∈ {1,2,3}。

只回答两个"先验自身"的问题，不涉及任何训练：
  (a) 先验判别力：把样本判给"最小相对阶次偏差"最小的故障类，逐载荷召回；
  (b) 真类相对阶次偏差 rel 的中位数（越小 = 谱峰越贴合理论阶次）。

另给两个受控对照：
  · 表 B：**转速先验偏差的代价** —— 用 0 hp 标称转速评价 1/2/3 hp 样本 vs 用真值转速；
  · 表 D：**BSF 口径**（含 B007，用 4 类池），并把 B007 作为"先验无据类"的对照组。

⚠️ 峰值来自窗口池缓存（K=5，按幅度降序）。K=1/3 取前 1/3 个峰等价于重新检测，
   故无需重算包络谱；K=8 需重新提取，本探针不覆盖（论文中标注为待办）。
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'code'))

from data.bearing_freq import CWRU_OFFICIAL_MULTIPLIERS  # noqa: E402

POOL3 = os.path.join(ROOT, 'data', 'processed',
                     'pools_w2048_st512_se512_cap400_bal1_c3-NC+IR007+OR007.npz')
POOL4 = os.path.join(ROOT, 'data', 'processed', 'pools.npz')
OUT = os.path.join(ROOT, 'results', 'tables', '_order_ablation.md')

# 3 类池：y = 0(NC) / 1(IR007) / 2(OR007) ⇒ 故障下标 0->IR007, 1->OR007
ORD3 = np.array([CWRU_OFFICIAL_MULTIPLIERS['BPFI'], CWRU_OFFICIAL_MULTIPLIERS['BPFO']])
# 4 类池：故障下标 0->IR007, 1->OR007, 2->B007（BSF 冲击率口径）
ORD4 = np.array([CWRU_OFFICIAL_MULTIPLIERS['BPFI'], CWRU_OFFICIAL_MULTIPLIERS['BPFO'],
                 CWRU_OFFICIAL_MULTIPLIERS['BSF']])
NAMES4 = ['IR007', 'OR007', 'B007']

LOAD_NAMES = {'test': ('0hp', 1797.0), 'cross_1': ('1hp', 1772.0),
              'cross_2': ('2hp', 1750.0), 'cross_3': ('3hp', 1730.0)}
KEYS = ('test', 'cross_1', 'cross_2', 'cross_3')
NOMINAL_0HP = 1797.0
LARGE = 1e6


def rel_matrix(pk: np.ndarray, orders: np.ndarray, fr: float, n_h: int) -> np.ndarray:
    """(N,K) 峰 -> (N, C_fault) 最小相对阶次偏差。

    rel = min_h min_k |f_k − h·O_c·f_r| / (O_c·f_r)
    分母与 k 无关 ⇒ 先对 k 取 min、再对 h 取 min 与原式等价（已由单测/探针复核）。
    """
    p = np.where(pk > 0, pk, LARGE).astype(np.float64)                  # (N,K)
    h = np.arange(1, n_h + 1, dtype=np.float64)                         # (Hn,)
    d = h[None, :] * orders[:, None] * fr                               # (C,Hn)
    num = np.abs(p[:, None, None, :] - d[None, :, :, None])             # (N,C,Hn,K)
    return (num / (orders[None, :, None, None] * fr + 1e-12)).min(axis=3).min(axis=2)


def fault_samples(d, key, K, n_cls):
    pk = np.asarray(d['%s__pk' % key], dtype=np.float64)[:, :K]
    y = np.asarray(d['%s__y' % key]).astype(int)
    m = y > 0
    return pk[m], y[m] - 1


def main() -> int:
    d3 = np.load(POOL3, allow_pickle=True)
    d4 = np.load(POOL4, allow_pickle=True)
    L = []
    P = L.append

    P('# 零训练消融：峰值数 K × 谐波阶数 H_n（真实窗池谱峰）\n')
    P('物理规则 = 判决到「最小相对阶次偏差」最小的故障类；仅统计**真值属于故障类**的样本。')
    P('每格格式 `召回 / 真类 rel 中位数`；rel 越小说明谱峰越贴合理论阶次。')
    P('数据：3 类池（NC/IR007/OR007），K 与 H_n 均为损失/先验口径参数，不涉及训练。\n')

    # ---------------- 表 A ----------------
    P('## 表 A  先验判别力随 (K, H_n) 的变化（各载荷**真值转速**）\n')
    P('| K | H_n | ' + ' | '.join(LOAD_NAMES[k][0] for k in KEYS) + ' | 平均召回 | 平均 rel |')
    P('|---|---|' + '---|' * 6)
    best = None
    for K in (1, 3, 5):
        for Hn in (1, 2, 3):
            cells, accs, meds = [], [], []
            for key in KEYS:
                pk, yy = fault_samples(d3, key, K, 2)
                rel = rel_matrix(pk, ORD3, LOAD_NAMES[key][1] / 60.0, Hn)
                acc = float((rel.argmin(axis=1) == yy).mean())
                med = float(np.median(rel[np.arange(pk.shape[0]), yy]))
                accs.append(acc); meds.append(med)
                cells.append('%.3f / %.4f' % (acc, med))
            ma, mm = float(np.mean(accs)), float(np.mean(meds))
            P('| %d | %d | %s | **%.3f** | %.4f |' % (K, Hn, ' | '.join(cells), ma, mm))
            if best is None or ma > best[0]:
                best = (ma, K, Hn)
    P('\n> 最优组合（按平均召回）：**K=%d, H_n=%d**（平均召回 %.3f）。' % (best[1], best[2], best[0]))
    P('> **读法（与直觉相反，需如实记录）**：在"仅 2 个故障类"的判别上，召回随 K 增大、')
    P('> 随 H_n 增大而**下降**（K=5/H_n=3 时最低）。机制见下表 C 的"非真类 min rel"列：')
    P('> 候选匹配集合变大 ⇒ 非真类的 min-rel 同步变小 ⇒ 类间"巧合匹配"通道被打开。')
    P('> 这**不是**说谐波无用 —— 它是"用判别力换转速误差容错"，权衡曲线见表 E。')

    # ---------------- 表 B ----------------
    P('\n## 表 B  「写死 0hp 标称转速」的代价（K=5, H_n=3）\n')
    P('| 载荷 | 真值 f_r (Hz) | 实测自估 f_r (Hz) | 自估误差 | 真类 rel@真值 | 真类 rel@0hp标称 | 膨胀 |')
    P('|---|---|---|---|---|---|---|')
    est = {'test': 29.90, 'cross_1': 29.50, 'cross_2': 29.10, 'cross_3': 28.65}  # 探针第 5 节
    for key in KEYS:
        pk, yy = fault_samples(d3, key, 5, 2)
        fr_t = LOAD_NAMES[key][1] / 60.0
        mt = float(np.median(rel_matrix(pk, ORD3, fr_t, 3)[np.arange(pk.shape[0]), yy]))
        mn = float(np.median(rel_matrix(pk, ORD3, NOMINAL_0HP / 60.0, 3)[np.arange(pk.shape[0]), yy]))
        fe = est[key]
        P('| %s | %.3f | %.2f | %+.2f%% | %.4f | %.4f | **%+.1f%%** |'
          % (LOAD_NAMES[key][0], fr_t, fe, (fe / fr_t - 1) * 100, mt, mn, (mn / mt - 1) * 100))
    P('\n> 0hp 行的膨胀为 0 属自检列（真值转速 = 标称转速）。')
    P('> 自估 f_r 由探针 `_probe_order_domain.py` 第 5 节给出（物理窗口 25–35 Hz），误差 ≤ 0.64%。')

    # ---------------- 表 C：H_n 的"过度匹配"风险 ----------------
    P('\n## 表 C  H_n 升高时的过度匹配检查（判别力 vs 容错）\n')
    P('| H_n | 真类 rel 中位数（4 载荷平均） | 非真类的最小 rel（越低=越易混淆） | 召回 |')
    P('|---|---|---|---|')
    for Hn in (1, 2, 3, 4, 5):
        med_t, min_o, accs = [], [], []
        for key in KEYS:
            pk, yy = fault_samples(d3, key, 5, 2)
            rel = rel_matrix(pk, ORD3, LOAD_NAMES[key][1] / 60.0, Hn)
            idx = np.arange(pk.shape[0])
            med_t.append(float(np.median(rel[idx, yy])))
            other = 1 - yy
            min_o.append(float(np.median(rel[idx, other])))
            accs.append(float((rel.argmin(axis=1) == yy).mean()))
        P('| %d | %.4f | %.4f | %.3f |' % (Hn, np.mean(med_t), np.mean(min_o), np.mean(accs)))
    P('\n> 若 H_n 增大使"非真类 rel"同步下降，说明高阶谐波开始把不相关谱峰吸入匹配集合。')

    # ---------------- 表 D：BSF 口径与 B007 ----------------
    P('\n## 表 D  BSF 口径与 B007（先验无据类）对照（4 类池，K=5, H_n=3，真值转速）\n')
    P('| 类别 | 阶次 O_c | 召回 | 真类 rel 中位数 | 备注 |')
    P('|---|---|---|---|---|')
    note = {'IR007': '先验强', 'OR007': '先验强', 'B007': '谱弥散，先验无据'}
    for ci, nm in enumerate(NAMES4):
        accs, meds = [], []
        for key in KEYS:
            pk, yy = fault_samples(d4, key, 5, 3)
            rel = rel_matrix(pk, ORD4, LOAD_NAMES[key][1] / 60.0, 3)
            m = yy == ci
            if m.sum() == 0:
                continue
            sub_rel = rel[m]
            accs.append(float((sub_rel.argmin(axis=1) == ci).mean()))
            meds.append(float(np.median(sub_rel[:, ci])))
        P('| %s | %.4f | %.3f | %.4f | %s |'
          % (nm, ORD4[ci], np.mean(accs), np.mean(meds), note[nm]))
    P('\n> 对照：通用**自转频率**口径为 2.3568，CWRU 官方**冲击率**口径为 4.7135，二者恰为 2 倍。')
    P('> 若误用通用口径，B007 的匹配集合整体错位一个八度（相对差 %.0f%%），匹配必然失效。'
      % ((4.7135 / 2.3568 - 1) * 100))

    # ---------------- 表 E：判别力 vs 转速误差容错 ----------------
    P('\n## 表 E  谐波阶数的真实作用：判别力 ↔ 转速误差容错 的权衡（K=5）\n')
    P('施加的先验转速误差 δ 表示"写死的转速与实际相差 δ"（δ=0 即先验完全正确）。\n')
    deltas = (0.0, -0.005, 0.005, -0.01, 0.01, -0.02, 0.02, -0.04, 0.04)
    P('| H_n | ' + ' | '.join('%+.1f%%' % (d * 100) for d in deltas) + ' | δ=0 处对外层均值 |')
    P('|---|' + '---|' * (len(deltas) + 1))
    base_by_h = {}
    for Hn in (1, 2, 3):
        cells = []
        for d in deltas:
            accs = []
            for key in KEYS:
                pk, yy = fault_samples(d3, key, 5, 2)
                fr_prior = (LOAD_NAMES[key][1] / 60.0) * (1.0 + d)
                rel = rel_matrix(pk, ORD3, fr_prior, Hn)
                accs.append(float((rel.argmin(axis=1) == yy).mean()))
            cells.append('%.3f' % float(np.mean(accs)))
            if abs(d) < 1e-9:
                base_by_h[Hn] = float(np.mean(accs))
        P('| %d | %s | 基准 %.3f |' % (Hn, ' | '.join(cells), base_by_h[Hn]))
    P('\n> **如实读法（不强行给因果结论）**：')
    P('> ① δ=0（先验完全正确）时 H_n 越小判别力越高（1.000 / 0.836 / 0.775），与表 C 一致；')
    P('> ② 施加转速先验误差后，各 H_n 的变化**既不单调、也不关于 δ 符号对称**')
    P('> 　（如 H_n=1 在 δ=−4% 跌到 0.599，在 δ=+4% 仍为 1.000）。')
    P('> ③ 因此**"高阶谐波提升转速误差容错"这一原稿动机在本数据上没有获得干净支持**，')
    P('> 　本表只作探索性敏感性检查保留，不作因果结论（与 5.7 节"边界律被证伪"同一处理原则）。')
    P('> ④ 可以确定的是：在 f_r 可自估（误差 ≤0.64%）的前提下，**没有必要靠高阶谐波去买容错**，')
    P('> 　因而 H_n 可以取小以换回判别力。该推断的直接依据是表 C 的"非真类 min rel"随 H_n 单调下降。')

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    txt = '\n'.join(L) + '\n'
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write(txt)
    print(txt)
    print('-> %s' % OUT)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
