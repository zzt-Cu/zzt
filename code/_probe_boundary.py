# -*- coding: utf-8 -*-
"""从已保存的 episode 级 logits **事后**重算"类别子集"边界 —— 不需要重训。

动机
----
诊断阶段已证明两条硬事实：
  1. IR/OR 的包络谱峰位**精确命中** BPFI/BPFO 谐波 → 频率先验在这两类上有效；
  2. CWRU 的 B007 包络谱**没有清晰的 BSF/2BSF 峰**，峰位中位数偏 14.6% → 先验在这类上无据可依。

要把它升级成**端到端模型层面**的论文证据，需要"同一批样本、同一噪声"下的三方对照：

    (a) 物理峰位规则（rel 最小者胜）  —— 先验本身能判对多少？
    (b) CNN-CE                        —— 纯数据学到多少？
    (c) CNN-Freq                      —— 加了物理约束之后是多少？

做法（零训练）
--------------
``evaluate.py`` 在**同工况留出集**上以 ``collect=True`` 保存了每个 query 的
``collect_logits / collect_y / collect_peak``（见 ``train.evaluate_episodes``）。
于是"限定在类别子集 S 上判决"可以**事后精确重算**：

    pred_S = argmax_{c in S} logits[:, c]
    acc_S  = mean(pred_S == y  |  y in S)

无需重训、无需改分类头。

类别数自适应
------------
脚本按每个 run 的 ``collect_logits.shape[1]`` 推断该 run 的类别数 ``n``，
类别名取 ``MASTER_CLASSES[:n]``。所以 4 类（--classes 缺省）与
3 类（``--classes NC,IR007,OR007``）的 run 可以放进同一份报告。

⚠️ 口径说明（必须写进论文局限性）
本脚本算的是 **"强制子集判决"**：模型训练时仍见过被删类，只是评估时不允许它选。
这与 **"真正只训 3 类"**（数据里根本不存在被删类）不是一回事，两者应当互为对照：

  - 本脚本（restricted argmax）= 便宜的第一读；
  - ``evaluate.py --classes NC,IR007,OR007`` 重训 = 正式复核。

两者同号则结论稳；不同号说明差异至少部分来自"见过但不许选"的抑制效应，
此时以重训结果为准。

用法
----
python _probe_boundary.py --glob "bnd4c_snrm8_lam5_*"
python _probe_boundary.py --glob "bnd3c_*" --classes NC,IR007,OR007
python _probe_boundary.py --out results/tables/_boundary_probe.md   # 相对路径以项目根为基准
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# 标签空间的**前缀顺序**：--classes 给出 N 个类时，标签就是 0..N-1 对应本列表前 N 项。
# 因此只要 evaluate.py 的 --classes 是本列表的子序列（如 NC,IR007,OR007），这里就自动对上。
MASTER_CLASSES: List[str] = ['NC', 'IR007', 'OR007', 'B007']

# 同工况留出集是 0hp ⇒ 用 0hp 标称转速做理论频率（与 evaluate.py 的 freq_consistency 同口径）
RPM_IN: float = 1797.0


# ----------------------------------------------------------------------
# 载入
# ----------------------------------------------------------------------
def _find_runs(logs_dir: str, pattern: str) -> List[str]:
    """返回匹配到的 run **目录**（不是 npz 文件本身）。

    ⚠️ 别直接把 npz 路径传下去：下游 ``_load_run`` 会再拼一次 'arrays.npz'，
    得到 ``.../arrays.npz/arrays.npz``，静默变成"没有可分析的 run"。
    """
    hits = sorted(glob.glob(os.path.join(logs_dir, pattern, 'arrays.npz')))
    return [os.path.dirname(h) for h in hits]


def _load_run(run_dir: str) -> Optional[Dict[str, object]]:
    p = os.path.join(run_dir, 'arrays.npz')
    if not os.path.exists(p):
        return None
    try:
        z = np.load(p, allow_pickle=False)
        if not all(k in z.files for k in ('collect_logits', 'collect_y')):
            return None
        logits = np.asarray(z['collect_logits'], dtype=np.float64)
        y = np.asarray(z['collect_y'], dtype=np.int64)
        peak = (np.asarray(z['collect_peak'], dtype=np.float64)
                if 'collect_peak' in z.files else None)
    except Exception as e:                       # 可能正在被写（后台实验运行中）
        print('!! 读取失败，跳过 %s：%s' % (os.path.basename(run_dir), e))
        return None
    meta: Dict[str, object] = {}
    rj = os.path.join(run_dir, 'result.json')
    if os.path.exists(rj):
        try:
            with open(rj, 'r', encoding='utf-8') as f:
                meta = json.load(f)
        except Exception:
            meta = {}
    return {'logits': logits, 'y': y, 'peak': peak, 'meta': meta,
            'run_id': os.path.basename(run_dir)}


def _parse_run_id(run_id: str) -> Tuple[str, int, str]:
    """从 run_id 里剥出 (method, seed, tag)。形如 '<tag>_<method>_seed<k>'。"""
    seed = -1
    if '_seed' in run_id:
        head, tail = run_id.rsplit('_seed', 1)
        try:
            seed = int(tail)
        except ValueError:
            seed = -1
        run_id = head
    method = '?'
    for m in ('CNN-Freq', 'CNN-CE', 'ProtoNet', 'MatchingNet', 'FineTune'):
        if run_id.endswith(m) or ('_' + m) in run_id:
            method = m
            run_id = run_id[:run_id.rfind(m)].rstrip('_')
            break
    return method, seed, run_id


# ----------------------------------------------------------------------
# 判决与统计
# ----------------------------------------------------------------------
def _subsets_for(n: int, classes: Sequence[str]) -> List[Tuple[str, List[int]]]:
    """全类 + 逐个"删掉一类"的子集。"""
    full = list(range(n))
    out: List[Tuple[str, List[int]]] = [('%dway' % n, full)]
    for i in range(n):
        out.append(('drop_%s' % classes[i], [j for j in full if j != i]))
    return out


def _restricted_acc(logits: np.ndarray, y: np.ndarray, subset: Sequence[int]) -> Dict[str, float]:
    """限定在 subset 上 argmax 的准确率，以及被保留的样本数。"""
    sub = np.asarray(sorted(subset), dtype=int)
    mask = np.isin(y, sub)
    if mask.sum() == 0:
        return {'acc': float('nan'), 'n': 0}
    lg = logits[np.ix_(mask, sub)]
    pred = sub[lg.argmax(axis=1)]
    return {'acc': float((pred == y[mask]).mean()), 'n': int(mask.sum())}


def _per_class_recall(logits: np.ndarray, y: np.ndarray, n: int) -> List[float]:
    pred = logits.argmax(axis=1)
    return [float((pred[y == c] == c).mean()) if (y == c).sum() else float('nan')
            for c in range(n)]


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> np.ndarray:
    """行=真类、列=预测类（与 train.confusion_matrix 同口径）。"""
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        if 0 <= t < n and 0 <= p < n:
            cm[t, p] += 1
    return cm


def _restricted_pred(logits: np.ndarray, y: np.ndarray, subset: Sequence[int],
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """在 subset 上 argmax，返回 (映射到子集内部下标的真标签, 预测下标)。

    ⚠️ **真标签也必须映射**到 `0..len(subset)-1`。否则真标签仍是原始类下标
    （例如 3 类下的 OR007=2），喂给 ``confusion_matrix(..., n=2)`` 时会因越界被静默丢弃，
    得到 "行全 0 / 某列全 1" 这种无意义矩阵 —— 这类错不报错，只让报告看起来"很确定"。
    """
    sub = np.asarray(sorted(subset), dtype=int)
    mask = np.isin(y, sub)
    lg = logits[np.ix_(mask, sub)]
    pred_local = lg.argmax(axis=1).astype(np.int64)
    lut = {int(v): i for i, v in enumerate(sub.tolist())}
    yt_local = np.asarray([lut[int(t)] for t in y[mask]], dtype=np.int64)
    return yt_local, pred_local


def _norm_confusion(cm: np.ndarray) -> np.ndarray:
    """按真类归一化（每行和为 1）；空行保持 0。"""
    s = cm.sum(axis=1, keepdims=True).astype(float)
    return np.divide(cm.astype(float), s, out=np.zeros_like(cm, dtype=float), where=s > 0)


# ----------------------------------------------------------------------
# 物理峰位规则（与 CNN 吃同一批噪声峰值）
# ----------------------------------------------------------------------
def _load_bearing_freq():
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    from data import bearing_freq as bf
    return bf


def _theory(bf, rpm: float, classes: Sequence[str],
            n_harmonics: int = 3, use_bsf_2x: bool = True):
    """返回 (H, f0)：类名 -> 谐波数组(Hz) / 基频(Hz)。NC 无理论频率，不在返回里。"""
    specs = bf.class_frequency_specs(rpm, classes, n_harmonics, use_bsf_2x)
    H, f0 = {}, {}
    for c in classes:
        s = specs[c]
        if s.harmonics:
            H[c] = np.asarray(s.harmonics, dtype=float)
            f0[c] = float(s.fundamental)
    return H, f0


def _phys_rule(peaks: np.ndarray, y: np.ndarray, classes: Sequence[str],
               H: Dict[str, np.ndarray], f0: Dict[str, float]) -> Optional[Dict[str, object]]:
    """峰位规则：rel[c] = min_{p,h} |p - H[h,c]| / f0[c]，判给 rel 最小的类。

    只统计**真类属于有理论频率的类**的样本（NC 无理论频率，不参与）。
    """
    fcs = [c for c in classes if c in H]
    if len(fcs) < 2:
        return None
    idx = [list(classes).index(c) for c in fcs]
    m = np.isin(y, idx)
    yy = y[m]
    pk_all = peaks[m]
    n = yy.size
    rel = np.full((n, len(fcs)), np.inf, dtype=float)
    for j, c in enumerate(fcs):
        h, f = H[c], f0[c]
        for i in range(n):
            pk = pk_all[i][pk_all[i] > 0]
            if pk.size == 0:
                continue
            rel[i, j] = float(np.min(np.abs(pk[:, None] - h[None, :]))) / f
    best = rel.argmin(axis=1)
    cm = _confusion_matrix(np.asarray([idx.index(int(t)) for t in yy], dtype=np.int64),
                           np.asarray(best, dtype=np.int64), len(fcs))
    recall, rel_true = {}, {}
    for j, c in enumerate(fcs):
        sel = (yy == idx[j])
        recall[c] = float((best[sel] == j).mean()) if sel.sum() else float('nan')
        rel_true[c] = float(np.median(rel[sel, j])) if sel.sum() else float('nan')
    return {'classes': fcs, 'confusion': cm, 'recall': recall,
            'rel_true_median': rel_true, 'n': int(n),
            'n_harmonics': int(H[fcs[0]].size)}


# ----------------------------------------------------------------------
# 逐 run 分析
# ----------------------------------------------------------------------
def analyze(run_dirs: Sequence[str], master: Sequence[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    bf = None
    for rd in run_dirs:
        d = _load_run(rd)
        if d is None:
            continue
        logits, y = d['logits'], d['y']      # type: ignore[assignment]
        peak = d['peak']
        n = int(logits.shape[1])
        if n > len(master):
            print('!! 跳过 %s：logits 有 %d 类，超出 MASTER_CLASSES(%d)'
                  % (d['run_id'], n, len(master)))
            continue
        classes = list(master)[:n]
        if y.size and (y.max() >= n or y.min() < 0):
            print('!! 跳过 %s：collect_y 范围 [%d,%d] 与 %d 类不符'
                  % (d['run_id'], y.min(), y.max(), n))
            continue
        method, seed, tag = _parse_run_id(str(d['run_id']))
        row: Dict[str, object] = {'run_id': d['run_id'], 'tag': tag, 'method': method,
                                  'seed': seed, 'n_class': n, 'n_query': int(y.size)}
        # 全类（应与 result.json 的 in_acc 一致 —— 作为代码正确性断言）
        full = _restricted_acc(logits, y, list(range(n)))
        row['acc_all'] = full['acc']
        jd = d['meta'].get('in_acc', float('nan')) if isinstance(d['meta'], dict) else float('nan')
        row['in_acc_json'] = jd
        row['acc_match'] = bool(abs(full['acc'] - jd) < 5e-3) if jd == jd else None
        for name, sub in _subsets_for(n, classes):
            row['acc_' + name] = _restricted_acc(logits, y, sub)['acc']
        for i, c in enumerate(classes):
            row['recall_' + c] = _per_class_recall(logits, y, n)[i]
        row['cm'] = _confusion_matrix(y, logits.argmax(axis=1), n)
        if isinstance(d['meta'], dict):
            row['freq_consistency'] = d['meta'].get('in_freq_consistency', float('nan'))
        # ---- 物理峰位规则：只用 (峰值, 标签)，同 tag 内各 run 共享同一噪声池 ----
        if peak is not None:
            try:
                if bf is None:
                    bf = _load_bearing_freq()
                H, f0 = _theory(bf, RPM_IN, classes, 3, True)
                row['phys'] = _phys_rule(np.asarray(peak, dtype=float), y, classes, H, f0)
                # CNN 限定在同样"有理论频率的类"上的混淆，便于逐格对照
                fcs = [c for c in classes if c in H]
                sub = [list(classes).index(c) for c in fcs]
                if len(sub) >= 2:
                    yt, yp = _restricted_pred(logits, y, sub)
                    row['cm_faults'] = _confusion_matrix(yt, yp, len(sub))
                    row['fault_classes'] = fcs
            except Exception as e:                              # pragma: no cover
                print('!! 物理规则计算失败（%s）：%s' % (d['run_id'], e))
        rows.append(row)
    return rows


# ----------------------------------------------------------------------
# 报告
# ----------------------------------------------------------------------
def _fmt(vals: Sequence[float], prec: int = 4) -> str:
    a = np.asarray([v for v in vals if v == v], dtype=float)
    if a.size == 0:
        return '-'
    if a.size == 1:
        return '%.*f' % (prec, a[0])
    return '%.*f±%.*f' % (prec, a.mean(), prec, a.std(ddof=0))


def _pairwise_delta(rs: List[Dict[str, object]], key: str) -> str:
    """CNN-Freq - CNN-CE 的**逐种子配对**差值（同 tag 内按 seed 对齐）。"""
    f = {int(r['seed']): float(r[key]) for r in rs
         if r['method'] == 'CNN-Freq' and key in r and r[key] == r[key]}
    c = {int(r['seed']): float(r[key]) for r in rs
         if r['method'] == 'CNN-CE' and key in r and r[key] == r[key]}
    seeds = sorted(set(f) & set(c))
    if not seeds:
        return '-'
    dd = [f[s] - c[s] for s in seeds]
    if len(dd) == 1:
        return '%+.4f' % dd[0]
    return '%+.4f±%.4f' % (float(np.mean(dd)), float(np.std(dd, ddof=0)))


def _md_table(L: List[str], header: Sequence[str], body: Sequence[Sequence[str]]) -> None:
    L.append('| ' + ' | '.join(header) + ' |')
    L.append('|' + '---|' * len(header))
    for b in body:
        L.append('| ' + ' | '.join(b) + ' |')


def _norm_mean_confusions(rows: Sequence[Dict[str, object]], key: str) -> Optional[np.ndarray]:
    cms = [np.asarray(r[key], dtype=float) for r in rows if key in r]
    if not cms:
        return None
    return np.mean([_norm_confusion(c) for c in cms], axis=0)


def report(rows: List[Dict[str, object]], master: Sequence[str]) -> str:
    L: List[str] = []
    L.append('# 边界性事后分析（restricted-argmax，零训练）\n')
    L.append('标签空间前缀顺序：%s\n' % ' / '.join('%d=%s' % (i, c) for i, c in enumerate(master)))
    L.append('口径：限定类别子集后 argmax，仅统计真类落在子集内的 query 样本。')
    L.append('⚠️ 这是"强制子集判决"（模型见过被删类），非"真正只训 3 类"。\n')

    bad = [r for r in rows if r.get('acc_match') is False]
    L.append('## 0. 代码正确性断言（acc_all 应 == result.json 的 in_acc）')
    if bad:
        L.append('❌ 有 %d 个 run 不一致，先别信下面的数：' % len(bad))
        for r in bad:
            L.append('  - %s: recomputed=%.4f json=%.4f'
                     % (r['run_id'], r['acc_all'], r['in_acc_json']))
    else:
        L.append('✅ 全部 %d 个 run 的 acc_all 与 result.json 一致（误差<5e-3）'
                 '→ 事后重算链路正确。' % len(rows))
    L.append('')

    by_tag: Dict[str, List[Dict[str, object]]] = {}
    for r in rows:
        by_tag.setdefault(str(r['tag']), []).append(r)

    for tag, rs in sorted(by_tag.items()):
        n = int(rs[0]['n_class'])
        classes = list(master)[:n]
        subsets = _subsets_for(n, classes)
        L.append('## tag = %s（%d 类，%d 个 run）' % (tag, n, len(rs)))
        L.append('| 子集 | CNN-CE | CNN-Freq | Δ(Freq-CE) |')
        L.append('|---|---|---|---|')
        for name, _ in subsets:
            k = 'acc_' + name
            ce = _fmt([float(r[k]) for r in rs if r['method'] == 'CNN-CE' and k in r and r[k] == r[k]])
            fr = _fmt([float(r[k]) for r in rs if r['method'] == 'CNN-Freq' and k in r and r[k] == r[k]])
            L.append('| %s | %s | %s | %s |' % (name, ce, fr, _pairwise_delta(rs, k)))
        L.append('')

        L.append('逐类召回（全类判决，均值）：')
        _md_table(L, ['类', 'CNN-CE', 'CNN-Freq'],
                  [[c,
                    _fmt([float(r['recall_' + c]) for r in rs if r['method'] == 'CNN-CE'
                          and 'recall_' + c in r and r['recall_' + c] == r['recall_' + c]]),
                    _fmt([float(r['recall_' + c]) for r in rs if r['method'] == 'CNN-Freq'
                          and 'recall_' + c in r and r['recall_' + c] == r['recall_' + c]])]
                   for c in classes])
        L.append('')

        k = 'freq_consistency'
        L.append('物理一致性 `in_freq_consistency`（预测为故障类者，谱峰落在理论 1–3 阶谐波 ±5% 内的比例）：')
        _md_table(L, ['', 'CNN-CE', 'CNN-Freq'],
                  [['freq_consistency',
                    _fmt([float(r[k]) for r in rs if r['method'] == 'CNN-CE' and k in r and r[k] == r[k]]),
                    _fmt([float(r[k]) for r in rs if r['method'] == 'CNN-Freq' and k in r and r[k] == r[k]])]])
        L.append('')

        for m in ('CNN-CE', 'CNN-Freq'):
            sub = [r for r in rs if r['method'] == m]
            ncm = _norm_mean_confusions(sub, 'cm')
            if ncm is None:
                continue
            L.append('%s 混淆矩阵（行=真类，列=预测类；按真类归一化；%d 个 run 均值）：' % (m, len(sub)))
            _md_table(L, ['true \\ pred'] + classes,
                      [['**%s**' % c] + ['%.3f' % ncm[i, j] for j in range(n)]
                       for i, c in enumerate(classes)])
            L.append('')

        # ---- 物理峰位规则 vs CNN：同一批 query 样本、同一噪声 ----
        phs = [r['phys'] for r in rs if r.get('phys')]
        if phs:
            fcs = list(phs[0]['classes'])          # type: ignore[arg-type]
            pcm = np.mean([_norm_confusion(np.asarray(p['confusion'], dtype=float))
                           for p in phs], axis=0)
            L.append('**物理峰位规则**（rel 最小者胜；%d 阶谐波；与上表**同一批噪声峰值**）'
                     '—— 行=真类，列=判给（%d 个 run 均值，每 run n=%d 故障样本）：'
                     % (phs[0]['n_harmonics'], len(phs), phs[0]['n']))
            _md_table(L, ['true \\ pred'] + fcs,
                      [['**%s**' % c] + ['%.3f' % pcm[i, j] for j in range(len(fcs))]
                       for i, c in enumerate(fcs)])
            L.append('')
            _md_table(L, ['指标'] + fcs, [
                ['物理规则召回（真类 = rel 最小）']
                + ['%.3f' % np.nanmean([p['recall'][c] for p in phs]) for c in fcs],
                ['真类 rel 中位数']
                + ['%.3f' % np.nanmean([p['rel_true_median'][c] for p in phs]) for c in fcs]])
            L.append('')

            cnn_cms = [r for r in rs if r['method'] == 'CNN-CE' and 'cm_faults' in r]
            raw_tot = (int(np.asarray(cnn_cms[0]['cm_faults']).sum()) if cnn_cms else 0)
            ncmf = _norm_mean_confusions(cnn_cms, 'cm_faults')
            if ncmf is not None and 'fault_classes' in cnn_cms[0]:
                fc2 = list(cnn_cms[0]['fault_classes'])   # type: ignore[arg-type]
                # 防线：真标签未映射到子集下标时会被 confusion_matrix 静默丢弃，
                # 表现为"总样本数为 0 / 某行全 0"。宁可显式报错，也不要画出假矩阵。
                if raw_tot == 0:
                    L.append('❌ `cm_faults` 总样本数为 0 —— 真标签未映射到子集下标，'
                             '本矩阵不可信，已跳过。（这是报告生成的 bug，不是模型问题）')
                    L.append('')
                    continue
                L.append('CNN-CE 限定在同样这些类上的混淆（restricted argmax，同一子集）'
                         '—— 行=真类，列=预测类（原始计数 %d）：' % raw_tot)
                _md_table(L, ['true \\ pred'] + fc2,
                          [['**%s**' % c] + ['%.3f' % ncmf[i, j] for j in range(len(fc2))]
                           for i, c in enumerate(fc2)])
                L.append('')
                # 逐类三方对照：先验能判对多少 / CNN 判对多少
                ce_rec = {c: np.nanmean([float(r['recall_' + c]) for r in rs
                                         if r['method'] == 'CNN-CE' and 'recall_' + c in r])
                          for c in fcs}
                fr_rec = {c: np.nanmean([float(r['recall_' + c]) for r in rs
                                         if r['method'] == 'CNN-Freq' and 'recall_' + c in r])
                          for c in fcs}
                L.append('逐类三方对照（先验上限 vs 实测）：')
                _md_table(L, ['类', '物理规则召回', '真类 rel 中位数', 'CNN-CE 召回', 'CNN-Freq 召回', '先验−CNN-CE'],
                          [[c,
                            '%.3f' % np.nanmean([p['recall'][c] for p in phs]),
                            '%.3f' % np.nanmean([p['rel_true_median'][c] for p in phs]),
                            '%.3f' % ce_rec[c], '%.3f' % fr_rec[c],
                            '%+.3f' % (np.nanmean([p['recall'][c] for p in phs]) - ce_rec[c])]
                           for c in fcs])
                L.append('')
                L.append('> 读法：`先验−CNN-CE` 显著为正 ⇒ **信息在数据里但被模型丢掉了**，'
                         '这类上优化一个正确构造的物理约束才有增益空间；')
                L.append('> 显著为负 ⇒ 该类的先验**无据可依**，强加频率约束只会把它推向错误类。')
                L.append('')

    L.append('## 附：逐 run 明细')
    keys = [k for k in rows[0] if k.startswith('acc_')] if rows else []
    header = ['run_id', 'method', 'seed', 'n_class'] + [k[len('acc_'):] for k in keys] + ['freq_cons']
    body = []
    for r in sorted(rows, key=lambda x: (str(x['tag']), str(x['method']), int(x['seed']))):
        fc = float(r.get('freq_consistency', float('nan')))
        body.append([str(r['run_id']), str(r['method']), str(r['seed']), str(r['n_class'])]
                    + ['%.4f' % r[k] for k in keys if k in r]
                    + ['%.4f' % fc if fc == fc else '-'])
    _md_table(L, header, body)
    L.append('')
    return '\n'.join(L)


def main(argv: Optional[Sequence[str]] = None) -> int:
    global MASTER_CLASSES, RPM_IN       # 必须在函数内任何引用之前声明
    ap = argparse.ArgumentParser('边界性事后分析（从保存的 logits 重算类别子集）')
    ap.add_argument('--logs', type=str, default=None, help='results/logs 目录（默认按项目根推断）')
    ap.add_argument('--glob', type=str, default='diag1shot_snrm8_*',
                    help='run 目录匹配模式（逗号分隔可传多个）')
    ap.add_argument('--classes', type=str, default=None,
                    help='标签空间前缀顺序，默认 %s。若某 run 只有 3 类，'
                         '取本列表前 3 项。' % ','.join(MASTER_CLASSES))
    ap.add_argument('--out', type=str, default=None, help='报告写到该路径（默认打印）')
    ap.add_argument('--rpm', type=float, default=None,
                    help='计算理论频率所用转速（默认 %g = 0hp 标称）' % RPM_IN)
    args = ap.parse_args(argv)

    if args.classes:
        MASTER_CLASSES = [x.strip() for x in args.classes.split(',') if x.strip()]
    if args.rpm is not None:
        RPM_IN = float(args.rpm)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logs = args.logs or os.path.join(root, 'results', 'logs')
    dirs: List[str] = []
    for p in [x.strip() for x in args.glob.split(',') if x.strip()]:
        dirs.extend(_find_runs(logs, p))
    seen = set()
    uniq = [d for d in dirs if not (d in seen or seen.add(d))]
    print('命中 %d 个 run：' % len(uniq))
    for d in uniq:
        print('  ' + os.path.relpath(d, logs))
    rows = analyze(uniq, MASTER_CLASSES)
    if not rows:
        print('!! 没有可分析的 run（arrays.npz 里缺 collect_logits/collect_y）')
        print('   提示：只有 collect=True 的同工况评估才会保存逐 query logits。')
        return 2
    text = report(rows, MASTER_CLASSES)
    if args.out:
        # 相对路径一律以**项目根 fault_diagnosis_pinn/** 为基准，并 normpath 归一化，
        # 否则 '..\\results\\...' 这类写法会静默写到项目外的目录。
        out = os.path.normpath(args.out if os.path.isabs(args.out)
                               else os.path.join(root, args.out))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w', encoding='utf-8') as f:
            f.write(text + '\n')
        print('\n报告已写入：%s' % out)
    else:
        print()
        print(text)
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
