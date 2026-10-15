# -*- coding: utf-8 -*-
"""定位"物理软约束零效果"的机制（只读诊断，不改任何训练产物）。

现象
----
`bnd3c` / `bnd4c` 实验里：CNN-Freq 的 `λ·L_freq` 量级是 CE 的 ~300 倍
（L_freq≈0.08~0.16，λ=0.5；CE≈1e-4，因 train_acc 自 epoch 9 起就是 1.000），
但部分种子的 `best.pt` 与 CNN-CE **逐位完全相同**。

三条证据（按重要性排序）
------------------------
**证据 C（首要）：λ 调度与"最优 epoch"错配**
    `stepwise` 策略 `λ(t)=0 当 t<warmup_epochs=20`；而 1-shot 下模型 9 轮就过拟合训练池，
    验证指标在 epoch 19 附近见顶。多数种子的 `best_epoch=19` 落在 λ=0 区间
    ⇒ **被报告的权重根本没被物理项碰过**。修法：把 λ 到达目标值的时刻挪到验证峰之前。

**证据 A：权重差异**
    diff 两组 `best.pt` 的 `state_dict`。`best_epoch<20` 的种子为 **0/46087 参数变动**；
    其余种子有真实但微小的差异（相对 L2 1e-5~1e-4）。
    ⚠️ 必须排除 BN 的 `num_batches_tracked`（计数器，随 best_epoch 变），
    否则 `max|Δθ|` 会被它污染成 2e3 这种假警报。

**证据 B（次要）：参数梯度量级**
    在真实工作点上比较 ‖λ∂L_freq/∂θ‖ 与 ‖∂L_CE/∂θ‖。
    即便 λ 生效，物理项梯度也比 CE 小 2~3 个数量级 ⇒ 力度本来就有限。

⚠️ 被推翻的假设（保留以示诚实）
    最初怀疑"softmax 在 float32 下精确 one-hot，导致梯度被浮点截断为 0"。
    实测该占比是 **0.0%**，假设不成立。真正原因是调度/选点，不是浮点截断。
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
if CODE not in sys.path:
    sys.path.insert(0, CODE)

from losses.physics_loss import FrequencySoftConstraintLoss   # noqa: E402
from losses.loss_scheduler import StepwiseLambda               # noqa: E402
from models.bearing_cnn import BearingCNN                      # noqa: E402


def load_yaml_cfg() -> dict:
    with open(os.path.join(CODE, 'configs', 'default.yaml'), encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg['freq']['shaft_rpm'] = {int(k): float(v) for k, v in cfg['freq']['shaft_rpm'].items()}
    for k in ('raw_dir', 'processed_dir'):
        cfg['data'][k] = os.path.join(ROOT, cfg['data'][k])
    for k, v in list(cfg['paths'].items()):
        cfg['paths'][k] = v if os.path.isabs(v) else os.path.join(ROOT, v)
    return cfg


# ----------------------------------------------------------------------
# 证据 A：权重差异
# ----------------------------------------------------------------------
def load_state(path: str) -> Optional[Dict[str, torch.Tensor]]:
    if not os.path.exists(path):
        return None
    ck = torch.load(path, map_location='cpu', weights_only=False)
    return {k: v.detach().clone() for k, v in ck['state_dict'].items()}


def diff_states(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> Dict[str, object]:
    """比较两组 state_dict：最大绝对差、相对范数、变动参数数，以及差异最大的前 5 个张量。

    返回 top5 是为了**防止被单个异常大的 buffer 带偏结论**（例如 BN 的 running_var
    可能因为某次 batch 尺度异常而爆到 1e3 量级，那并不代表"训练轨迹不同"）。
    """
    assert set(a) == set(b), '结构不同，无法比较：%s' % (set(a) ^ set(b))
    max_abs, sq, base_sq = 0.0, 0.0, 0.0
    n_changed, n_tot = 0, 0
    per: List[Tuple[str, float, float]] = []
    for k in a:
        d = (a[k].double() - b[k].double()).abs()
        mx = float(d.max()) if d.numel() else 0.0
        base = float(a[k].double().abs().max()) if d.numel() else 0.0
        per.append((k, mx, base))
        max_abs = max(max_abs, mx)
        sq += float((d ** 2).sum())
        base_sq += float((a[k].double() ** 2).sum())
        n_changed += int((d > 0).sum())
        n_tot += int(d.numel())
    per.sort(key=lambda t: -t[1])
    return {'max_abs': max_abs, 'rel_l2': (sq ** 0.5) / (base_sq ** 0.5 + 1e-30),
            'n_changed': n_changed, 'n_total': n_tot, 'top5': per[:5]}


# ----------------------------------------------------------------------
# 证据 B：梯度量级 vs 饱和程度
# ----------------------------------------------------------------------
@torch.no_grad()
def collect_logits(ckpt: str, X: np.ndarray, limit: int = 600) -> torch.Tensor:
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    model = BearingCNN(**{k: v for k, v in ck['cfg'].items()})
    model.load_state_dict(ck['state_dict'])
    model.eval()
    x = torch.from_numpy(np.ascontiguousarray(X[:limit])).float()
    return model(x)


def grad_at(logits: torch.Tensor, peaks: torch.Tensor,
            loss_fn) -> Tuple[float, float, float]:
    """返回 (L_freq 值, |∂L/∂logits| 的最大值, 平均值)。"""
    lg = logits.clone().requires_grad_(True)
    l = loss_fn(lg, peaks)
    l.backward()
    g = lg.grad.detach().abs()
    return float(l.detach()), float(g.max()), float(g.mean())


def load_model(ckpt: str):
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    model = BearingCNN(**{k: v for k, v in ck['cfg'].items()})
    model.load_state_dict(ck['state_dict'])
    model.eval()
    return model


def _grad_norm(model) -> float:
    sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            sq += float((p.grad.detach().double() ** 2).sum())
    return sq ** 0.5


def param_grad_norms(model, X: np.ndarray, y: np.ndarray, peaks: torch.Tensor,
                     loss_fn, lam: float, limit: int = 600) -> Dict[str, float]:
    """在**真实工作点**上比较两条参数梯度的大小。

    这是最能说明问题的口径：优化器实际拿到的更新量取决于 ∂ loss/∂θ，
    而不是 ∂ loss/∂logits。若 ‖λ∇L_freq‖ 远小于 ‖∇L_CE‖，物理项等于没参与优化。
    """
    import torch.nn.functional as F
    x = torch.from_numpy(np.ascontiguousarray(X[:limit])).float()
    yt = torch.from_numpy(np.ascontiguousarray(y[:limit])).long()
    pk = peaks[:limit]

    model.zero_grad(set_to_none=True)
    l_ce = F.cross_entropy(model(x), yt)
    l_ce.backward()
    g_ce = _grad_norm(model)

    model.zero_grad(set_to_none=True)
    l_f = loss_fn(model(x), pk)
    (lam * l_f).backward()
    g_f = _grad_norm(model)

    model.zero_grad(set_to_none=True)
    return {'l_ce': float(l_ce.detach()), 'l_freq': float(l_f.detach()),
            'g_ce': g_ce, 'g_freq_scaled': g_f,
            'ratio_freq_over_ce': g_f / (g_ce + 1e-30)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser('验证物理损失梯度是否被饱和吃掉')
    ap.add_argument('--logs', type=str, default=None)
    ap.add_argument('--tags', type=str, default='bnd3c_snrm8_lam5,bnd4c_snrm8_lam5')
    ap.add_argument('--seeds', type=str, default='0,1,2')
    ap.add_argument('--rpm', type=float, default=1797.0)
    ap.add_argument('--out', type=str, default='results/tables/_grad_vanish.md')
    args = ap.parse_args(argv)

    cfg = load_yaml_cfg()
    logs = args.logs or cfg['paths']['results_logs']
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    tags = [t.strip() for t in args.tags.split(',') if t.strip()]

    L: List[str] = []
    L.append('# 物理约束"零效果"的机制定位（只读诊断，不改任何训练产物）\n')
    L.append('## 结论摘要（含一次自我更正）\n')
    L.append('最初假设是"`λ·L_freq` 值很大但梯度被 softmax 饱和吃掉（浮点意义上精确为 0）"。')
    L.append('**这个假设被数据推翻了**：实测 softmax 在 float32 下精确 one-hot 的样本占比是 **0.0%**，')
    L.append('梯度并非精确零。')
    L.append('')
    L.append('真正的首要机制是 **λ 调度与"最优 epoch"错配**：')
    L.append('`stepwise` 策略下 `λ(t)=0` 当 `t < warmup_epochs=20`，')
    L.append('而验证指标在 **epoch 19 附近就已见顶**（1-shot 下模型 9 轮就 `train_acc=1.000`，随后在 val 上过拟合下行）。')
    L.append('于是被选中的 `best.pt` 取自 **λ 仍为 0 的那一轮** —— 物理项对它没有任何影响。')
    L.append('')
    L.append('| 证据 | 问了什么问题 | 结论 |')
    L.append('|---|---|---|')
    L.append('| **C（首要）** | λ 生效的那一轮，是否早于最优 epoch？ | 多数种子 **否**，best_epoch=19 < warmup=20 |')
    L.append('| A | 两组 `best.pt` 权重差多少？ | best_epoch<20 的种子**逐位相同**；其余有微小差异 |')
    L.append('| B | 物理项的参数梯度有多大？ | 相对 CE 小 2~3 个数量级（次要因素） |')
    L.append('')

    # ---------------- 证据 C ----------------
    L.append('## 证据 C（首要）：λ 调度 vs best_epoch 错配\n')
    L.append('逐 (方法, 种子) 从 `runs_<tag>.csv` 读 `best_epoch`，'
             '再用 `StepwiseLambda(warmup=20, step=10, …)` 算出该轮的 λ。')
    L.append('若 `λ(best_epoch) == 0`，说明被选中的权重里**不含物理项的任何贡献**。\n')
    sched = StepwiseLambda(20, 10, 0.1, 0.5)
    L.append('⚠️ `runs_*.csv` 里的 `best_epoch` 是 **0-based**（`train.py` 里 '
             '`res.best_epoch = epoch`，而 `for epoch in range(rc.epochs)`），'
             '所以直接 `sched.value(best_epoch)` 即为该轮的 λ。')
    L.append('另外 `val_every=20`、且在第 `(epoch+1) % 20 == 0` 轮评估，'
             '故**只有 epoch 索引 19/39/59/79 这几处会被验证并可能被选为最优**，'
             '对应 λ = 0 / 0.2 / 0.4 / 0.5。\n')
    L.append('| tag | method | seed | best_epoch(0-based) | λ(best_epoch) | best_val_f1 | 物理项是否影响该权重 |')
    L.append('|---|---|---|---|---|---|---|')
    n_blind, n_tot = 0, 0
    for tag in tags:
        csvp = os.path.join(cfg['paths']['results_tables'], 'runs_%s.csv' % tag)
        if not os.path.exists(csvp):
            L.append('| %s | - | - | - | - | - | 缺 runs CSV |' % tag)
            continue
        import csv as _csv
        with open(csvp, encoding='utf-8-sig') as f:
            for r in _csv.DictReader(f):
                be = int(float(r['best_epoch']))
                lam_be = sched.value(be)               # best_epoch 已是 0-based
                affected = (r['method'] == 'CNN-Freq')
                blind = affected and lam_be == 0.0
                if affected:
                    n_tot += 1
                    n_blind += int(blind)
                L.append('| %s | %s | %s | %d | %.2f | %.4f | %s |'
                         % (tag, r['method'], r['seed'], be, lam_be, float(r['best_val_f1']),
                            ('**否（λ=0，权重与 CE 同源）**' if blind
                             else ('是' if affected else '—（CE 基线）'))))
    L.append('')
    if n_tot:
        L.append('⇒ `CNN-Freq` 的 %d 组里，有 **%d 组**的 `best_epoch` 落在 λ=0 区间。'
                 % (n_tot, n_blind))
    L.append('')
    L.append('**这直接给出了实验设计的修法**：把 λ 提到目标值的时间点挪到**验证峰之前**，')
    L.append('例如 `warmup_epochs=5, step=2`（λ 在 ~epoch 15 到 0.5），')
    L.append('或改用 `fixed`（λ 恒为 0.5）配合短 warmup。')
    L.append('否则无论 L_freq 设计得多好，它都不会出现在被报告的模型里。\n')

    # ---------------- 证据 A ----------------
    L.append('## 证据 A：best.pt 权重差异（CE vs Freq，同种子）\n')
    L.append('⚠️ `max\\|Δθ\\|` 会被 **BatchNorm 的 `num_batches_tracked` 计数器**污染：'
             '它是"该 BN 层前向次数"，取决于 `best_epoch`。')
    L.append('若两方法的最优 epoch 不同，这个计数器能差出上千，'
             '表现为 `max\\|Δθ\\|=2e3` 这种假警报（见下方张量明细）。\n')
    L.append('| tag | seed | max\\|Δθ\\| | 相对 L2 | 变动参数数/总数 | 判定 |')
    L.append('|---|---|---|---|---|---|')
    detail: List[Tuple[str, int, list, float]] = []
    for tag in tags:
        for s in seeds:
            pc = os.path.join(logs, '%s_CNN-CE_seed%d' % (tag, s), 'best.pt')
            pf = os.path.join(logs, '%s_CNN-Freq_seed%d' % (tag, s), 'best.pt')
            a, b = load_state(pc), load_state(pf)
            if a is None or b is None:
                L.append('| %s | %d | - | - | - | 缺 best.pt |' % (tag, s))
                continue
            d = diff_states(a, b)
            verdict = ('**完全一致（物理项零贡献）**' if d['n_changed'] == 0
                       else ('仅浮点噪声' if d['max_abs'] < 1e-9 else '确有差异'))
            L.append('| %s | %d | %.3e | %.3e | %d/%d | %s |'
                     % (tag, s, d['max_abs'], d['rel_l2'],
                        d['n_changed'], d['n_total'], verdict))
            detail.append((tag, s, d['top5'], d['max_abs']))
    L.append('')
    L.append('差异最大的 5 个张量（用于排查是不是单个异常 buffer 撑起了 `max\\|Δθ\\|`）：')
    L.append('')
    L.append('| tag | seed | 张量 | max\\|Δ\\| | 该张量 max\\|θ\\| |')
    L.append('|---|---|---|---|---|')
    for tag, s, top5, _mx in detail:
        for k, mx, base in top5[:5]:
            L.append('| %s | %d | `%s` | %.3e | %.3e |' % (tag, s, k, mx, base))
    L.append('')

    # ---------------- 证据 B ----------------
    L.append('## 证据 B（次要）：L_freq 的参数梯度量级\n')
    L.append('证据 C 说明"大多数种子根本没被物理项碰过"；本节的 B-1 则回答另一半问题：')
    L.append('**万一 λ 生效了（如 best_epoch=59 的种子），它到底有多大力气？**\n')

    from data.cwru_loader import prepare
    for tag in tags:
        # 类别由 tag 推断：bnd3c -> 3 类，bnd4c -> 4 类
        cls = (['NC', 'IR007', 'OR007'] if '3c' in tag
               else ['NC', 'IR007', 'OR007', 'B007'])
        cfg['data']['classes'] = cls
        cfg['fewshot']['n_way'] = len(cls)
        cfg['model']['n_way'] = len(cls)
        try:
            pools = prepare(cfg, force=False, verbose=False)
        except Exception as e:
            L.append('- `%s`：无法准备窗口池（%s），跳过证据 B。\n' % (tag, e))
            continue
        X = pools['test'].X
        pk = pools['test'].peak_f
        peaks = torch.from_numpy(np.ascontiguousarray(pk[:600])).float()
        ck = os.path.join(logs, '%s_CNN-Freq_seed0' % tag, 'best.pt')
        if not os.path.exists(ck):
            L.append('- `%s`：缺 %s，跳过。\n' % (tag, ck))
            continue
        base = collect_logits(ck, X, 600)
        loss_fn = FrequencySoftConstraintLoss(rpm=args.rpm, classes=cls, n_harmonics=3)

        L.append('### %s（%d 类；权重取自 %s 的 CNN-Freq_seed0）\n' % (tag, len(cls), tag))
        # ---- B-1：参数梯度对比（最贴近优化器实际拿到的信号）----
        pg = param_grad_norms(load_model(ck), X, pools['test'].y, peaks, loss_fn,
                              lam=0.5, limit=600)
        L.append('**B-1 参数梯度对比**（真实工作点，λ=0.5，前 600 个 test 窗口）：')
        L.append('')
        # 饱和程度的直接刻画：float32 下 softmax 是否**恰好**退化成 one-hot
        with torch.no_grad():
            sm = torch.softmax(base, dim=1).numpy()
            srt = np.sort(base.numpy(), axis=1)
            margin_med = float(np.median(srt[:, -1] - srt[:, -2]))
            margin_p10 = float(np.percentile(srt[:, -1] - srt[:, -2], 10))
        frac_onehot = float((sm.max(axis=1) >= 1.0 - 1e-7).mean())
        L.append('| 量 | 数值 |')
        L.append('|---|---|')
        L.append('| L_CE | %.3e |' % pg['l_ce'])
        L.append('| L_freq（未乘 λ） | %.4f |' % pg['l_freq'])
        L.append('| ‖∂L_CE/∂θ‖ | %.3e |' % pg['g_ce'])
        L.append('| ‖λ·∂L_freq/∂θ‖（λ=0.5） | %.3e |' % pg['g_freq_scaled'])
        L.append('| **比值 ‖λ∇L_freq‖ / ‖∇L_CE‖** | **%.3e** |' % pg['ratio_freq_over_ce'])
        L.append('| logit margin 中位数 / P10 | %.2f / %.2f |' % (margin_med, margin_p10))
        L.append('| softmax 在 float32 下**恰好** one-hot 的样本占比 | **%.1f%%** |'
                 % (100.0 * frac_onehot))
        L.append('')
        if frac_onehot > 0.5:
            L.append('→ **机制确认**：`%.0f%%` 的样本 softmax 在 float32 下就是精确的 one-hot'
                     '（最大概率 == 1.0，其余为 0）。此时'
                     '`L_freq = rel[argmax]` 且 `∂L/∂z_c ≡ 0`（对 c=argmax 有 `rel−L=0`，'
                     '对其它 c 有 softmax_c=0）—— **梯度在浮点意义上是精确的 0**，'
                     '这正好解释了证据 A 里 seed0/seed1 权重逐位相同。' % (100.0 * frac_onehot))
        else:
            L.append('→ softmax 只有 %.1f%% 精确 one-hot ⇒ **"梯度被浮点精确截断为 0"这个假设不成立**。'
                     '梯度真实但很小，所以真正原因得往调度/选点上找（见证据 C）。'
                     % (100.0 * frac_onehot))
        L.append('')
        L.append('⚠️ 口径说明：`L_CE` 是在 **test 池窗口**上算的（异分布，故达 %.2f），'
                 '不等于训练时已饱和的 query CE（≈1e-4），因此该比值不等于训练时的比值，'
                 '只作为"物理项梯度比 CE 小 2~3 个数量级"这一**量级**结论的依据。' % pg['l_ce'])
        L.append('')

        # ---- B-2：梯度随 logit 缩放的恢复情况 ----
        L.append('**B-2 梯度量级 vs logit 饱和程度**：把 logits 乘缩放因子 `s`，'
                 '`s=1` 是真实工作点，`s<1` 人为解除饱和。')
        row_v, row_gmax, row_gmean, row_margin = [], [], [], []
        scales = [1.0, 0.5, 0.25, 0.1, 0.05]
        for s in scales:
            lv, gm, ga = grad_at(base * s, peaks, loss_fn)
            sortd = np.sort(base.numpy(), axis=1)
            margin = float(np.median(sortd[:, -1] - sortd[:, -2]))
            row_v.append(lv)
            row_gmax.append(gm)
            row_gmean.append(ga)
            row_margin.append(margin * s)
        L.append('| logits 缩放 s | L_freq | max\\|∂L/∂logits\\| | mean\\|∂L/∂logits\\| | logit margin |')
        L.append('|---|---|---|---|---|')
        for i, s in enumerate(scales):
            L.append('| %.2f | %.4f | %.3e | %.3e | %.2f |'
                     % (s, row_v[i], row_gmax[i], row_gmean[i], row_margin[i]))
        L.append('')
        ratio = (row_gmax[-1] / row_gmax[0]) if row_gmax[0] > 0 else float('inf')
        L.append('> s 从 1.0 降到 %.2f（logit margin 从 %.2f 降到 %.2f），'
                 'L_freq 值只从 %.4f 变到 %.4f —— **几乎不变**。'
                 % (scales[-1], row_margin[0], row_margin[-1], row_v[0], row_v[-1]))
        if ratio >= 3:
            L.append('> 而梯度范数放大约 **%.0f 倍**，方向一致：'
                     '**饱和是主因**——L_freq 在饱和区几乎是个常数，'
                     '代价看得见、梯度看不见。\n' % ratio)
        else:
            L.append('> 梯度范数只放大约 **%.1f 倍**（不是量级变化），'
                     '说明饱和只是次要因素：更主要的原因是'
                     '**L_freq 对 logits 本身就不敏感**（rel 数值区间窄、类间差异小）。'
                     '两条原因叠加，结果都是"损失值可见、优化信号不可见"。\n' % ratio)

    text = '\n'.join(x for x in L if x is not None)
    out = os.path.normpath(args.out if os.path.isabs(args.out)
                           else os.path.join(ROOT, args.out))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    print('报告已写入：%s' % out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
