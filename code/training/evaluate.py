# -*- coding: utf-8 -*-
"""评估与统计检验（1000 episodes × 5 次独立运行，均值 ± 95% 置信区间）。

统计口径（刻意分开报告，避免"把相关样本当独立样本"把 CI 算窄）
----------------------------------------------------------------
1. **seed 级**：每次独立运行得到 1 个 episode 均值 → n=5 个样本 → 均值 ± 95% CI(t 分布)。
   这是最保守、也最诚实的口径，**论文主表用这个**。
2. **episode 级**：把 5×1000 个 episode 的准确率当作样本 → n=5000。
   ⚠️ 同一池内不同 episode 高度相关（窗口复用），这个 CI 会**偏乐观**，
   只作为参考一并列出，并在 6.9 局限性中说明。
3. **窗口级（低复用对照）**：对 CNN-CE / CNN-Freq 这类直接分类头，
   在留出池的**每一个不同窗口**上跑一次，得到不依赖 episode 采样的准确率与 macro-F1。
   用于交叉验证"episode 级结论不是采样 artifact"。

配对 t 检验
-----------
CNN-Freq vs CNN-CE 同时给两个 p 值：
- ``p_seed``    : 配对在 5 个种子上（n=5，功效低，但样本独立）；
- ``p_episode`` : 配对在共同的 5000 个 episode 上（n=5000，样本相关，p 会偏小）。
两个方向都报，由读者判断，而不是只挑好看的那个。

额外诊断
--------
``freq_consistency``：预测类别的理论特征频率（含 1–3 阶谐波）与实际谱峰
在容差内匹配的 query 样本占比。它是**物理先验是否真的被学到**的直接证据，
也用于跨工况分析（转速变化后一致性是否下降）与失败案例分析。
"""
from __future__ import annotations

import math
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    from ..data import bearing_freq as bf
    from ..data.cwru_loader import WindowPool, NOMINAL_RPM
    from ..losses.physics_loss import FrequencySoftConstraintLoss, build_h_matrix
    from ..models.bearing_cnn import BearingCNN
    from ..utils import logger as lg
    from ..utils.seed import derive_seed, setup_seed
    from .episode_sampler import EpisodeSampler
    from .train import (METHODS, MethodRunner, RunConfig, TrainResult,
                        confusion_matrix, evaluate_episodes, fit, load_cfg,
                        macro_f1, resolve_device)
except (ImportError, ValueError):                                      # pragma: no cover
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data import bearing_freq as bf
    from data.cwru_loader import WindowPool, NOMINAL_RPM
    from losses.physics_loss import FrequencySoftConstraintLoss, build_h_matrix
    from models.bearing_cnn import BearingCNN
    from utils import logger as lg
    from utils.seed import derive_seed, setup_seed
    from training.episode_sampler import EpisodeSampler
    from training.train import (METHODS, MethodRunner, RunConfig, TrainResult,
                                confusion_matrix, evaluate_episodes, fit, load_cfg,
                                macro_f1, resolve_device)


# ----------------------------------------------------------------------
# 统计工具
# ----------------------------------------------------------------------
def mean_ci(values: Sequence[float], ci: float = 0.95) -> Dict[str, float]:
    """样本均值与 t 分布 95% 置信区间半宽。n<2 时区间置 nan。"""
    a = np.asarray([v for v in values if v == v], dtype=np.float64)
    n = a.size
    if n == 0:
        return {'mean': float('nan'), 'std': float('nan'), 'ci_half': float('nan'), 'n': 0}
    m = float(a.mean())
    if n < 2:
        return {'mean': m, 'std': 0.0, 'ci_half': float('nan'), 'n': n}
    sd = float(a.std(ddof=1))
    try:
        from scipy import stats
        tcrit = float(stats.t.ppf(0.5 + ci / 2.0, df=n - 1))
    except Exception:                                                  # pragma: no cover
        tcrit = 1.96
    return {'mean': m, 'std': sd, 'ci_half': tcrit * sd / math.sqrt(n), 'n': n}


def paired_ttest(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """配对 t 检验（a - b）。返回 t、p、自由度、均值差。"""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        k = min(x.size, y.size)
        x, y = x[:k], y[:k]
    d = x - y
    if d.size < 2:
        return {'t': float('nan'), 'p': float('nan'), 'df': 0, 'mean_diff': float('nan')}
    try:
        from scipy import stats
        t, p = stats.ttest_rel(x, y)
        return {'t': float(t), 'p': float(p), 'df': int(d.size - 1),
                'mean_diff': float(d.mean()), 'std_diff': float(d.std(ddof=1)),
                'n_pairs': int(d.size)}
    except Exception:                                                  # pragma: no cover
        return {'t': float('nan'), 'p': float('nan'), 'df': int(d.size - 1),
                'mean_diff': float(d.mean()), 'n_pairs': int(d.size)}


# ----------------------------------------------------------------------
# 物理一致性诊断
# ----------------------------------------------------------------------
@torch.no_grad()
def freq_consistency(model: BearingCNN, runner: MethodRunner, sampler: EpisodeSampler,
                     rpm: float, n_episodes: int, classes: Sequence[str],
                     n_harmonics: int = 3, tol: float = 0.05,
                     device: str = 'cpu') -> Dict[str, float]:
    """统计"预测类别的理论谐波与实测谱峰在相对容差 tol 内匹配"的 query 占比。

    对 NC（无理论频率）单独给出占比，不与故障类混在一起平均。
    """
    H, f0, valid = build_h_matrix(rpm, list(classes), n_harmonics)
    H = H.numpy(); f0 = f0.numpy(); valid = valid.numpy()
    model.eval()
    hit, tot = 0, 0
    hit_fault, tot_fault = 0, 0
    hit_nc, tot_nc = 0, 0
    for ep in sampler.episodes(n_episodes):
        ep = ep.to(device)
        pred = runner.predict(model, ep).argmax(dim=1).cpu().numpy()
        peak = ep.query_peak.cpu().numpy()
        for i in range(peak.shape[0]):
            c = int(pred[i])
            pk = peak[i][peak[i] > 0]
            tot += 1
            if not valid[c]:
                tot_nc += 1
                continue
            tot_fault += 1
            if pk.size == 0:
                continue
            rel = np.abs(pk[:, None] - H[:, c][None, :]).min(axis=1) / (f0[c] + 1e-12)
            if rel.min() <= tol:
                hit += 1
                hit_fault += 1
    model.train(False)
    return {
        'freq_consistency_all': hit / max(1, tot),
        'freq_consistency_fault': hit_fault / max(1, tot_fault),
        'n_query': tot, 'n_pred_fault': tot_fault, 'n_pred_nc': tot_nc,
        'tol': tol, 'rpm': rpm,
    }


# ----------------------------------------------------------------------
# 窗口级（低复用）评估：只适用于带直接分类头的方法
# ----------------------------------------------------------------------
@torch.no_grad()
def evaluate_windows_direct(model: BearingCNN, pool: WindowPool, n_way: int = 4,
                            batch: int = 256, device: str = 'cpu') -> Dict[str, object]:
    """在池内**每一个不同窗口**上各跑一次（不经过 episode 采样）。"""
    model.eval()
    X = torch.from_numpy(np.ascontiguousarray(pool.X)).to(device)
    logits = []
    for i in range(0, X.shape[0], batch):
        logits.append(model(X[i:i + batch]).cpu())
    logits = torch.cat(logits, dim=0).numpy()
    pred = logits.argmax(axis=1)
    y = pool.y
    return {'n_windows': int(y.size), 'acc': float((pred == y).mean()),
            'macro_f1': macro_f1(y, pred, n_way),
            'confusion_matrix': confusion_matrix(y, pred, n_way).tolist(),
            'pred': pred.tolist()}


# ----------------------------------------------------------------------
# 一次完整实验：训练 + 同工况测试 + 跨工况测试
# ----------------------------------------------------------------------
def train_and_eval(cfg: dict, method: str, seed: int, pools: Dict[str, WindowPool],
                   rc: RunConfig, out_dir: Optional[str] = None,
                   verbose: bool = True) -> Dict[str, object]:
    """训练一次，在 0hp 留出集与 1/2/3hp 目标域上评估。返回一行扁平结果。"""
    device = resolve_device(rc.device)
    t0 = time.time()
    res = fit(cfg, pools['train'], pools['val'], method, rc, out_dir=out_dir, verbose=verbose)
    model = res._model
    runner = MethodRunner(method, _freq_loss_for(cfg), cfg, rc)

    classes = list(cfg['data']['classes'])
    n_way = rc.n_way
    row: Dict[str, object] = {
        'method': method, 'seed': seed, 'n_params': res.n_params,
        'flops_m': res.flops_m, 'best_epoch': res.best_epoch,
        'best_val_f1': res.best_val_f1, 'best_val_acc': res.best_val_acc,
        'train_time_s': res.train_time_s, 'epoch_time_s': res.epoch_time_s,
        'peak_gpu_mem_mb': res.peak_gpu_mem_mb, 'cfg_hash': res.cfg_hash,
    }

    # ---- 同工况（0hp）测试 ----
    sampler = EpisodeSampler.from_cfg(pools['test'], cfg, seed=derive_seed(seed, 'test'),
                                      device=device, name='test')
    v = evaluate_episodes(model, runner, sampler, int(cfg['eval']['test_episodes']), n_way,
                          collect=True, device=device)
    row.update({'in_acc': v['acc'], 'in_acc_std': v['acc_std'],
                'in_f1': v['f1'], 'in_f1_std': v['f1_std']})
    row['_in_acc_per_episode'] = v['acc_per_episode']
    row['_in_f1_per_episode'] = v['f1_per_episode']
    row['_in_confusion'] = v['confusion_matrix']
    row['_collect'] = v['collect']

    # 窗口级对照（仅直接分类头）
    if method in ('CNN-CE', 'CNN-Freq'):
        w = evaluate_windows_direct(model, pools['test'], n_way, device=device)
        row['in_win_acc'] = w['acc']
        row['in_win_f1'] = w['macro_f1']
        row['_in_win_confusion'] = w['confusion_matrix']
    row['in_freq_consistency'] = freq_consistency(
        model, runner, sampler, NOMINAL_RPM[int(cfg['data']['train_loads'][0])],
        200, classes, device=device)['freq_consistency_fault']

    # ---- 跨工况（0hp -> 1/2/3 hp） ----
    for hp in cfg['data']['cross_loads']:
        hp = int(hp)
        key = 'cross_%d' % hp
        if key not in pools:
            continue
        s = EpisodeSampler.from_cfg(pools[key], cfg, seed=derive_seed(seed, 'cross', hp),
                                    device=device, name=key)
        cv = evaluate_episodes(model, runner, s, 400, n_way, collect=False, device=device)
        row['x%d_acc' % hp] = cv['acc']
        row['x%d_f1' % hp] = cv['f1']
        row['_x%d_confusion' % hp] = cv['confusion_matrix']
        # 目标域用自身标称转速（现场假设：装了转速表）
        row['x%d_freq_consistency' % hp] = freq_consistency(
            model, runner, s, NOMINAL_RPM[hp], 200, classes, device=device)['freq_consistency_fault']
    row['total_time_s'] = time.time() - t0
    return row


def _freq_loss_for(cfg: dict) -> FrequencySoftConstraintLoss:
    """构造频率软约束损失实例（用模块级已导入的类，兼容包内/脚本两种运行方式）。"""
    return FrequencySoftConstraintLoss.from_cfg(cfg)


# ----------------------------------------------------------------------
# 多方法 × 多种子
# ----------------------------------------------------------------------
def run_experiments(cfg: dict, methods: Sequence[str], seeds: Sequence[int],
                    rc_factory, pools: Dict[str, WindowPool], tables_dir: str,
                    logs_dir: str, tag: str = 'main', verbose: bool = True,
                    resume: bool = False
                    ) -> List[Dict[str, object]]:
    """跑 (方法 × 种子) 全组合，落盘 per-run CSV 与聚合 CSV。

    resume=True 时，若某 (方法, 种子) 的 out_dir 下已有 result.json，则直接读回
    已完成的指标、跳过重训——用于长时间批次被中断后安全续跑。
    注意：读回的 row 不含以 '_' 开头的大数组（per-episode 序列等），因此
    episode 级配对检验在跨批次续跑时会自动降级为只做 seed 级检验。
    """
    rows: List[Dict[str, object]] = []
    for method in methods:
        for seed in seeds:
            run_id = '%s_%s_seed%d' % (tag, method, seed)
            out_dir = os.path.join(logs_dir, run_id)
            if resume:
                rj = os.path.join(out_dir, 'result.json')
                if os.path.isfile(rj):
                    try:
                        prev = lg.load_json(rj)
                        if isinstance(prev, dict) and 'in_acc' in prev:
                            prev = dict(prev)
                            prev['run_id'] = run_id
                            rows.append(prev)
                            if verbose:
                                print('\n>>> [%s] seed=%d  (%s)  [resume: 读回已有 result.json]'
                                      % (method, seed, run_id), flush=True)
                            continue
                    except Exception as e:  # 读不回来就重跑，不静默丢实验
                        print('  [resume] 读取 %s 失败（%s），改为重跑' % (rj, e))
            rc = rc_factory(method, seed)
            if verbose:
                print('\n>>> [%s] seed=%d  (%s)' % (method, seed, run_id))
            row = train_and_eval(cfg, method, seed, pools, rc, out_dir=out_dir, verbose=verbose)
            row['run_id'] = run_id
            rows.append(row)
            # 每个 run 立刻落盘（防止后续崩溃丢失已完成的实验）
            lg.write_json(os.path.join(out_dir, 'result.json'),
                          {k: v for k, v in row.items() if not k.startswith('_')})
            _save_run_arrays(out_dir, row)
    _aggregate_and_save(rows, tables_dir, tag, cfg)
    return rows


def _save_run_arrays(out_dir: str, row: Dict[str, object]) -> None:
    """把大数组（per-episode 准确率、混淆矩阵、t-SNE 特征）单独存 npz，避免塞进 CSV。"""
    arrs = {}
    for k, v in row.items():
        if k.startswith('_') and isinstance(v, (np.ndarray, list, dict)):
            if k == '_collect':
                for kk, vv in v.items():
                    arrs['collect_%s' % kk] = np.asarray(vv)
            else:
                arrs[k[1:]] = np.asarray(v)
    if arrs:
        np.savez_compressed(os.path.join(out_dir, 'arrays.npz'), **arrs)


def _aggregate_and_save(rows: List[Dict[str, object]], tables_dir: str, tag: str,
                        cfg: dict) -> None:
    """按方法聚合：均值 ± 95%CI、标准差，以及 CNN-Freq vs CNN-CE 的配对 t 检验。"""
    from collections import OrderedDict
    by_method: 'OrderedDict[str, List[Dict[str, object]]]' = OrderedDict()
    for r in rows:
        by_method.setdefault(str(r['method']), []).append(r)

    metrics = ['in_acc', 'in_f1', 'in_win_acc', 'in_win_f1',
               'x1_acc', 'x2_acc', 'x3_acc', 'x1_f1', 'x2_f1', 'x3_f1',
               'train_time_s', 'epoch_time_s', 'peak_gpu_mem_mb',
               'in_freq_consistency', 'x1_freq_consistency',
               'x2_freq_consistency', 'x3_freq_consistency']
    agg_rows = []
    for m, rs in by_method.items():
        out = {'method': m, 'n_seeds': len(rs)}
        for k in metrics:
            vals = [float(r[k]) for r in rs if k in r and r[k] == r[k]]
            if not vals:
                continue
            st = mean_ci(vals, ci=0.95)
            out['%s_mean' % k] = st['mean']
            out['%s_std' % k] = st['std']
            out['%s_ci95' % k] = st['ci_half']
        agg_rows.append(out)
    lg.write_csv(os.path.join(tables_dir, 'results_%s.csv' % tag), agg_rows)

    # per-run 明细
    flat = [{k: v for k, v in r.items() if not k.startswith('_')} for r in rows]
    lg.write_csv(os.path.join(tables_dir, 'runs_%s.csv' % tag), flat)

    # ---- 配对 t 检验 ----
    tests = []
    if 'CNN-Freq' in by_method and 'CNN-CE' in by_method:
        a, b = by_method['CNN-Freq'], by_method['CNN-CE']
        for scope, key in [('in-domain(0hp)', 'in_acc'), ('cross 1hp', 'x1_acc'),
                           ('cross 2hp', 'x2_acc'), ('cross 3hp', 'x3_acc')]:
            va = [float(r[key]) for r in a if key in r]
            vb = [float(r[key]) for r in b if key in r]
            st = paired_ttest(va, vb)
            row = {'scope': scope, 'metric': key, 'n_seeds': st.get('n_pairs', 0),
                   'mean_CNN-Freq': float(np.mean(va)) if va else float('nan'),
                   'mean_CNN-CE': float(np.mean(vb)) if vb else float('nan'),
                   'delta': st.get('mean_diff', float('nan')),
                   't': st.get('t', float('nan')), 'p_seed': st.get('p', float('nan')),
                   'df': st.get('df', 0)}
            # episode 级配对（共同 episode，样本相关，p 偏小，仅供参考）
            ea = np.concatenate([np.asarray(r['_in_acc_per_episode']) for r in a
                                 if '_in_acc_per_episode' in r]) if key == 'in_acc' else None
            eb = np.concatenate([np.asarray(r['_in_acc_per_episode']) for r in b
                                 if '_in_acc_per_episode' in r]) if key == 'in_acc' else None
            if ea is not None and eb is not None and ea.size == eb.size:
                st2 = paired_ttest(ea, eb)
                row['p_episode'] = st2.get('p', float('nan'))
                row['n_episode_pairs'] = st2.get('n_pairs', 0)
            tests.append(row)
    if tests:
        lg.write_csv(os.path.join(tables_dir, 'ttest_%s.csv' % tag), tests)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _split_csv(s: str) -> List[str]:
    return [x.strip() for x in s.split(',') if x.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser('evaluate: few-shot 对比实验')
    ap.add_argument('--methods', type=str, default=','.join(METHODS))
    ap.add_argument('--seeds', type=str, default='0,1,2,3,4')
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--episodes-per-epoch', type=int, default=None)
    ap.add_argument('--val-episodes', type=int, default=None)
    ap.add_argument('--test-episodes', type=int, default=None)
    ap.add_argument('--val-every', type=int, default=None)
    ap.add_argument('--device', type=str, default=None)
    ap.add_argument('--tag', type=str, default='main')
    ap.add_argument('--outdir', type=str, default=None)
    ap.add_argument('--n-shot', type=int, default=None, dest='n_shot',
                    help='覆盖 fewshot.n_shot（做 shot 数敏感性/极端小样本实验用）')
    ap.add_argument('--n-query-per-class', type=int, default=None, dest='nq',
                    help='覆盖 fewshot.n_query_per_class')
    ap.add_argument('--disjoint-support-query', action='store_true', dest='disjoint_sq',
                    help='支持集与查询集【互斥】采样（合并后无放回抽取再切分）。'
                         '默认关闭 = 历史行为（有放回、可能重叠），用于重叠效应的敏感性对照。')
    ap.add_argument('--window', type=int, default=None,
                    help='覆盖 data.window（改变窗长会触发窗口池重建）')
    ap.add_argument('--classes', type=str, default=None, dest='classes',
                    help='覆盖 data.classes，逗号分隔（如 NC,IR007,OR007）。'
                         '**标签 = 类别列表下标**，故会同步改写 fewshot.n_way / model.n_way，'
                         '并让窗口池缓存按类别集合重新取指纹（不会误复用 4 类池）。')
    ap.add_argument('--snr-db', type=float, default=None, dest='snr_db',
                    help='在**时域**按该 SNR(dB) 加噪后重新做包络特征化（鲁棒性实验）')
    # ---- λ 调度覆盖 ----
    # 动机（实测教训）：默认 warmup_epochs=20 会让 λ 直到 epoch 索引 20 才 >0，
    # 而 1-shot 下模型 9 轮就 train_acc=1.000，验证指标在 epoch 索引 19 就见顶。
    # 于是 best.pt 取自 **λ 仍为 0** 的那一轮，物理项对最终模型毫无影响
    # （实测：部分种子 CE 与 Freq 的权重逐位相同）。要做"物理项到底有没有用"的实验，
    # 必须把 λ 到达目标值的时刻挪到验证峰之前。
    ap.add_argument('--lambda-strategy', type=str, default=None, dest='lam_strategy',
                    choices=['stepwise', 'fixed', 'adaptive', 'zero'],
                    help='覆盖 loss.lambda_strategy')
    ap.add_argument('--warmup-epochs', type=int, default=None, dest='warmup',
                    help='覆盖 loss.warmup_epochs（建议做 1-shot 时设 4~5）')
    ap.add_argument('--lambda-step', type=int, default=None, dest='lam_step',
                    help='覆盖 loss.step（λ 每多少轮加一档）')
    ap.add_argument('--lambda-max', type=float, default=None, dest='lam_max',
                    help='覆盖 loss.lambda_max')
    ap.add_argument('--fixed-lambda', type=float, default=None, dest='fixed_lam',
                    help='覆盖 loss.fixed_lambda（配合 --lambda-strategy fixed）')
    # ---- 随机频率对照（消融）----
    # 动机：判断物理约束的增益到底来自"物理"，还是来自"多一项损失=正则化"。
    # 设 freq_jitter>0 后，每个类的基频被乘 U(1-j,1+j) 后重建谐波表；
    # 约束强度/形式/超参/λ 调度全部不动，只有频率的物理正确性被破坏。
    ap.add_argument('--freq-jitter', type=float, default=None, dest='freq_jitter',
                    help='随机频率对照：基频乘 U(1-j,1+j)（如 0.5）。0=关闭（默认用真实频率）')
    ap.add_argument('--jitter-seed', type=int, default=None, dest='jitter_seed',
                    help='随机频率表的 seed（固定值 ⇒ 各转速共用同一张随机表）')
    # ---- NC 屏蔽对照（消融）----
    # 动机：NC ∉ V，其 logit 不参与 softmax，但 per_sample 仍对全部 query 求平均，
    # 于是真值为 NC 的样本也收到"推向 rel 最小故障类"的错误监督。
    # 打开后只对真值 ≠ NC 的样本求平均，用于检验该近似是否影响结论。
    ap.add_argument('--mask-nc', action='store_true', dest='mask_nc',
                    help='L_order 只对真值≠NC 的 query 求平均（默认关闭，保持原口径）')
    ap.add_argument('--mask-nc-rescale', action='store_true', dest='mask_nc_rescale',
                    help='配合 --mask-nc：再乘 N_F/N 抵消有效梯度放大，用于分离混杂')
    # ---- 转速自适应 / 阶次域（论文升级项）----
    ap.add_argument('--speed-mode', type=str, default=None, dest='speed_mode',
                    choices=['fixed', 'estimate', 'estimate_decoupled', 'jitter'],
                    help="转速先验来源：fixed=用标称转速（旧做法）；"
                         "estimate=由谱峰自估 f_r（免转速表）；jitter=f_r 区间随机化（鲁棒训练）；"
                         "estimate_decoupled=同 estimate 的估计器，但每 episode 建表用的 f_r "
                         "取自上一个 episode（滞后一批），用于端到端解耦干预（3.7 节数据耦合机制检验）")
    ap.add_argument('--speed-jitter-rpm', type=float, default=None, dest='speed_jitter_rpm',
                    help='speed-mode=jitter 时的 f_r 随机半宽（rpm，如 100 表示 ±100 rpm）')
    ap.add_argument('--rpm-window', type=str, default=None, dest='rpm_window',
                    help='自估 f_r 的物理可行窗口（如 "1500,2100"）；默认 1500,2100')
    ap.add_argument('--rpm-override', type=float, default=None, dest='rpm_override',
                    help='强制覆盖**训练期**标称转速（用于"转速先验偏差"敏感性实验，'
                         '如设为 1900 模拟先验偏 +5.7%%）')
    ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--resume', action='store_true', dest='resume',
                    help='断点续跑：已有 result.json 的 (方法,种子) 直接读回，不重训')
    args = ap.parse_args(argv)

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cfg = load_cfg(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'configs', 'default.yaml'))
    cfg['freq']['shaft_rpm'] = {int(k): float(v) for k, v in cfg['freq']['shaft_rpm'].items()}
    # 路径统一以项目根 fault_diagnosis_pinn/ 为基准（YAML 中即为相对路径）
    for k in ('raw_dir', 'processed_dir'):
        cfg['data'][k] = os.path.join(root, cfg['data'][k])
    for k, v in list(cfg['paths'].items()):
        cfg['paths'][k] = v if os.path.isabs(v) else os.path.join(root, v)
    if args.outdir:
        # 在 results/ 下开一层子目录，把本次运行的图表/表格/日志隔离
        sub = args.outdir
        cfg['paths']['results_logs'] = os.path.join(cfg['paths']['results_logs'], sub)
        cfg['paths']['results_tables'] = os.path.join(cfg['paths']['results_tables'], sub)
        cfg['paths']['results_figures'] = os.path.join(cfg['paths']['results_figures'], sub)
        cfg['paths']['checkpoints'] = os.path.join(cfg['paths']['checkpoints'], sub)
    lg.ensure_dirs(cfg['paths']['results_logs'], cfg['paths']['results_tables'],
                   cfg['paths']['results_figures'], cfg['paths']['checkpoints'])
    if args.epochs is not None:
        cfg['train']['epochs'] = args.epochs
    if args.episodes_per_epoch is not None:
        cfg['train']['episodes_per_epoch'] = args.episodes_per_epoch
    if args.val_episodes is not None:
        cfg['train']['val_episodes'] = args.val_episodes
    if args.test_episodes is not None:
        cfg['eval']['test_episodes'] = args.test_episodes
    if args.val_every is not None:
        cfg['train']['val_every'] = args.val_every
    if args.n_shot is not None:
        cfg['fewshot']['n_shot'] = args.n_shot
    if args.nq is not None:
        cfg['fewshot']['n_query_per_class'] = args.nq
    if args.disjoint_sq:
        cfg['fewshot']['disjoint_support_query'] = True
    if args.window is not None:
        cfg['data']['window'] = args.window
    if args.lam_strategy is not None:
        cfg['loss']['lambda_strategy'] = args.lam_strategy
    if args.warmup is not None:
        cfg['loss']['warmup_epochs'] = args.warmup
    if args.lam_step is not None:
        cfg['loss']['step'] = args.lam_step
    if args.lam_max is not None:
        cfg['loss']['lambda_max'] = args.lam_max
    if args.fixed_lam is not None:
        cfg['loss']['fixed_lambda'] = args.fixed_lam
    if args.freq_jitter is not None:
        cfg['loss']['freq_jitter'] = float(args.freq_jitter)
    if args.jitter_seed is not None:
        cfg['loss']['jitter_seed'] = int(args.jitter_seed)
    if getattr(args, 'mask_nc', False):
        cfg['loss']['mask_nc'] = True
    if getattr(args, 'mask_nc_rescale', False):
        cfg['loss']['mask_nc_rescale'] = True
    if args.speed_mode is not None:
        cfg['loss']['speed_mode'] = str(args.speed_mode)
    if args.speed_jitter_rpm is not None:
        cfg['loss']['speed_jitter_rpm'] = float(args.speed_jitter_rpm)
    if args.rpm_window is not None:
        w = [float(x) for x in _split_csv(args.rpm_window)]
        if len(w) != 2 or w[0] >= w[1]:
            raise SystemExit('--rpm-window 需为 "lo,hi" 且 lo<hi，收到 %s' % args.rpm_window)
        cfg['loss']['rpm_window'] = w
    if args.rpm_override is not None:
        # 只改**训练域**（train_loads[0]）的标称转速，模拟"先验写死的转速是错的"。
        hp0 = int(cfg['data']['train_loads'][0])
        cfg['freq']['shaft_rpm'][hp0] = float(args.rpm_override)

    from data.cwru_loader import FILE_CODES, prepare, add_noise_to_pools

    # ---- 类别集合覆盖（--classes）----
    # 标签 = 类别列表下标 ⇒ 改集合就是改标签空间。必须同时同步：
    #   ① cfg['data']['classes']  —— 建池用（并进入缓存指纹）
    #   ② cfg['fewshot']['n_way'] —— episode 采样用
    #   ③ cfg['model']['n_way']   —— 分类头 FC2 维度用
    # 漏掉任一都会出现"FC2 输出 4 维但标签只有 3 类"这类静默错位。
    if args.classes:
        cls = _split_csv(args.classes)
        if len(set(cls)) != len(cls):
            raise SystemExit('--classes 存在重复项：%s' % cls)
        unknown = [c for c in cls if c not in FILE_CODES]
        if unknown:
            raise SystemExit('未知类别 %s；FILE_CODES 中可选：%s'
                             % (unknown, sorted(FILE_CODES)))
        cfg['data']['classes'] = cls
        cfg['data']['label_map'] = {c: i for i, c in enumerate(cls)}
        cfg['fewshot']['n_way'] = len(cls)
        cfg['model']['n_way'] = len(cls)
    else:
        # YAML 里的 label_map 只是文档性质，以 classes 顺序为准，运行时统一重算
        cfg['data']['label_map'] = {c: i for i, c in enumerate(cfg['data']['classes'])}

    pools = prepare(cfg, force=False, verbose=not args.quiet)
    if args.snr_db is not None:
        from data.envelope_spectrum import EnvelopeConfig
        print('\n--- 时域加噪 SNR=%.1f dB，并重新做包络特征化 ---' % args.snr_db)
        pools = add_noise_to_pools(pools, args.snr_db, 12345,
                                   EnvelopeConfig.from_cfg(cfg), verbose=not args.quiet)
        cfg['data']['_noisy_snr_db'] = args.snr_db

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    def rc_factory(method: str, seed: int) -> RunConfig:
        t = cfg['train']
        return RunConfig(
            method=method, seed=int(seed), epochs=int(t['epochs']),
            episodes_per_epoch=int(t['episodes_per_epoch']),
            val_every=int(t.get('val_every', 10)), val_episodes=int(t['val_episodes']),
            lr=float(t['lr']), weight_decay=float(t['weight_decay']),
            grad_clip=float(t.get('grad_clip', 5.0)), device=device,
            n_way=int(cfg['fewshot']['n_way']), finetune_steps=30, finetune_lr=1e-3,
            progress=not args.quiet)

    run_id = lg.new_run_id(args.tag)
    log_path = lg.run_log_path(cfg['paths']['results_logs'], run_id)
    with lg.Tee(log_path):
        print('=== 运行 %s ===' % run_id)
        print('methods =', _split_csv(args.methods))
        print('seeds   =', _split_csv(args.seeds))
        print('device  =', device)
        print('classes = %s  (n_way=%d, n_shot=%d, n_query/class=%d, support=%d)'
              % (cfg['data']['classes'], int(cfg['fewshot']['n_way']),
                 int(cfg['fewshot']['n_shot']), int(cfg['fewshot']['n_query_per_class']),
                 int(cfg['fewshot']['n_way']) * int(cfg['fewshot']['n_shot'])))
        print('epochs=%d  episodes/epoch=%d  val_every=%d  test_episodes=%d'
              % (cfg['train']['epochs'], cfg['train']['episodes_per_epoch'],
                 cfg['train'].get('val_every', 10), cfg['eval']['test_episodes']))
        print('cfg_hash =', lg.cfg_hash(cfg))
        from losses.loss_scheduler import build_lambda_scheduler
        _s = build_lambda_scheduler(cfg['loss'])
        print('lambda   = %s   λ(t) 前 25 轮 = %s'
              % (_s.describe(), ['%.2f' % x for x in _s.values(25)]))
        _j = float(cfg['loss'].get('freq_jitter', 0.0))
        print('freq先验 = %s'
              % ('**随机频率对照** jitter=±%.0f%%, seed=%d（真实频率已被破坏，用于消融）'
                 % (100.0 * _j, int(cfg['loss'].get('jitter_seed', 0)))
                 if _j > 0 else '真实理论特征频率（BPFO/BPFI/BSF 及谐波）'))
        _sm = str(cfg['loss'].get('speed_mode', 'fixed'))
        _sjt = float(cfg['loss'].get('speed_jitter_rpm', 0.0))
        _rw = cfg['loss'].get('rpm_window', None)
        print('转速先验 = mode=%s  train_rpm=%.1f%s%s'
              % (_sm, cfg['freq']['shaft_rpm'][int(cfg['data']['train_loads'][0])],
                 ('  jitter=±%.0f rpm' % _sjt) if _sm == 'jitter' else '',
                 ('  window=%s rpm（自估，免转速表）' % _rw)
                 if _sm in ('estimate', 'estimate_decoupled') else '')
              + ('  [解耦干预：f_r 滞后一批]' if _sm == 'estimate_decoupled' else ''))
        for k, v in pools.items():
            print('  pool %-9s n=%-6d %s' % (k, len(v), v.n_per_class))
        rows = run_experiments(cfg, _split_csv(args.methods),
                               [int(s) for s in _split_csv(args.seeds)],
                               rc_factory, pools, cfg['paths']['results_tables'],
                               cfg['paths']['results_logs'], tag=args.tag,
                               verbose=not args.quiet,
                               resume=bool(getattr(args, 'resume', False)))
        print('\n=== 汇总（均值 ± 95%%CI，n=%d seeds）===' % len(_split_csv(args.seeds)))
        print('%-12s %-18s %-18s %-18s' % ('method', 'in-domain(0hp)', 'cross 2hp', 'in win acc'))
        for r in rows:
            pass
        import collections
        by = collections.OrderedDict()
        for r in rows:
            by.setdefault(r['method'], []).append(r)
        for m, rs in by.items():
            def ms(k):
                v = [float(x[k]) for x in rs if k in x]
                if not v:
                    return '-'
                s = mean_ci(v)
                return '%.4f±%.4f' % (s['mean'], s['ci_half'])
            print('%-12s %-18s %-18s %-18s' % (m, ms('in_acc'), ms('x2_acc'), ms('in_win_acc')))
    print('\n日志: %s' % log_path)
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
