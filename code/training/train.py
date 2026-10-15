# -*- coding: utf-8 -*-
"""训练主流程：episodic few-shot 训练 + 5 个对比方法的统一实现。

方法族（全部**共用同一个 45,732 参数的 1D-CNN 骨干**，只有"头 + 损失"不同）
----------------------------------------------------------------------------
====================  ==================================================  ==============
方法                  训练/推理方式                                        物理约束
====================  ==================================================  ==============
``CNN-CE``            直接 4 分类头 + 交叉熵（基线）                      无
``CNN-Freq``          同上 + ``λ(t)·L_freq``（**本文方法**）               频率软约束
``ProtoNet``          支持集 embedding 求类原型，按负欧氏距离分类         无
``MatchingNet``       查询-支持余弦相似度注意力，加权支持标签             无
``FineTune``          同 CNN-CE 训练；**测试时**在 20 个支持样本上微调     无
====================  ==================================================  ==============

公平性（对应论文 6.1 节"实验设置"）
-----------------------------------
- 骨干完全相同（同结构、同参数量、同初始化分布）；
- 相同 episode 采样器、相同 episode 预算、相同优化器（AdamW）、
  相同学习率（2e-4）、相同调度（Cosine Annealing）、相同 epoch 数；
- 相同的早停/选优准则（验证集 macro-F1）；
- 除 CNN-Freq 额外使用物理损失外，其余超参一致；FineTune 的额外微调步数是该方法的
  固有组成部分，其步数与学习率一并报告。

性能与可复现
------------
- 窗口池常驻显存，episode 只采样索引，host↔device 通信趋近于零；
- 验证用**固定种子的 100 个 episode**，保证不同 epoch 之间可比、模型选择无偏；
- 每个 run 落盘 ``history.csv`` / ``env.json`` / ``best.pt`` / ``summary.json``。
"""
from __future__ import annotations

import copy
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ..data.cwru_loader import WindowPool
    from ..losses.loss_scheduler import build_lambda_scheduler
    from ..losses.physics_loss import FrequencySoftConstraintLoss
    from ..models.bearing_cnn import BearingCNN, count_parameters, estimate_flops
    from ..utils import logger as lg
    from ..utils.seed import derive_seed, setup_seed
    from .episode_sampler import Episode, EpisodeSampler
except (ImportError, ValueError):                                      # pragma: no cover
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.cwru_loader import WindowPool
    from losses.loss_scheduler import build_lambda_scheduler
    from losses.physics_loss import FrequencySoftConstraintLoss
    from models.bearing_cnn import BearingCNN, count_parameters, estimate_flops
    from utils import logger as lg
    from utils.seed import derive_seed, setup_seed
    from training.episode_sampler import Episode, EpisodeSampler

METHODS: List[str] = ['CNN-CE', 'CNN-Freq', 'ProtoNet', 'MatchingNet', 'FineTune']
METHOD_USES_FREQ: Dict[str, bool] = {m: (m == 'CNN-Freq') for m in METHODS}
METHOD_IS_METRIC: Dict[str, bool] = {'ProtoNet': True, 'MatchingNet': True}


# ----------------------------------------------------------------------
# 指标
# ----------------------------------------------------------------------
def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """宏平均 F1（对本项目尤其重要：CWRU 类别样本数天然失衡，
    只看 accuracy 会被样本多的正常类刷高）。"""
    f1s = []
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1s))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    m = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            m[t, p] += 1
    return m


# ----------------------------------------------------------------------
# 方法运行器
# ----------------------------------------------------------------------
@dataclass
class RunConfig:
    method: str = 'CNN-Freq'
    seed: int = 42
    epochs: int = 200
    episodes_per_epoch: int = 100
    val_every: int = 10
    val_episodes: int = 100
    lr: float = 2e-4
    weight_decay: float = 0.01
    grad_clip: float = 5.0
    device: str = 'auto'
    finetune_steps: int = 30
    finetune_lr: float = 1e-3
    matching_scale: float = 10.0
    n_way: int = 4
    progress: bool = True


def resolve_device(device: str) -> str:
    if device != 'auto':
        return device
    return 'cuda' if torch.cuda.is_available() else 'cpu'


class MethodRunner:
    """把"方法"抽象成"训练一步"与"预测"两个动作。"""

    def __init__(self, method: str, freq_loss: FrequencySoftConstraintLoss,
                 cfg: dict, rc: RunConfig):
        if method not in METHODS:
            raise ValueError('未知方法 %s，可选 %s' % (method, METHODS))
        self.method = method
        self.freq_loss = freq_loss
        self.cfg = cfg
        self.rc = rc
        self.n_way = rc.n_way
        # ---- NC 屏蔽对照（消融；默认关闭 ⇒ 与既有结果逐位一致）----
        # 动机：NC ∉ V，其 logit 不参与 softmax，但 per_sample 仍对**全部** query 求平均，
        # 于是真值为 NC 的样本也收到"推向 rel 最小故障类"的错误监督（论文 §2.3 已披露）。
        # 打开后：只对真值 ≠ NC 的样本求平均，NC 样本彻底不参与 L_order。
        self.mask_nc = bool(cfg.get('loss', {}).get('mask_nc', False))
        self.mask_nc_rescale = bool(cfg.get('loss', {}).get('mask_nc_rescale', False))
        _cls = list(cfg.get('data', {}).get('classes', []))
        self.nc_index = _cls.index('NC') if 'NC' in _cls else 0

    # ------------------------------------------------------------------
    def _embed(self, model: BearingCNN, x: torch.Tensor) -> torch.Tensor:
        return model.embed(x)

    def _prototypes(self, e_s: torch.Tensor, sy: torch.Tensor) -> torch.Tensor:
        protos = []
        for c in range(self.n_way):
            m = (sy == c)
            if m.sum() == 0:
                protos.append(torch.zeros_like(e_s[0]))
            else:
                protos.append(e_s[m].mean(dim=0))
        return torch.stack(protos, dim=0)                       # (n_way, D)

    # ------------------------------------------------------------------
    def train_step(self, model: BearingCNN, ep: Episode, lam: float
                   ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """返回 (loss, stats)。"""
        stats: Dict[str, float] = {'lambda': float(lam)}
        if self.method in ('CNN-CE', 'CNN-Freq'):
            logits = model(ep.query_x)
            l_ce = F.cross_entropy(logits, ep.query_y)
            stats['loss_ce'] = float(l_ce.detach())
            loss = l_ce
            if self.method == 'CNN-Freq' and lam > 0:
                if self.mask_nc:
                    # 只对真值 ≠ NC 的样本求平均：用 return_parts 取逐样本项后自行归约。
                    # mask_nc=False 时走下面原路径，数值与既有实验逐位一致。
                    # mask_nc_rescale=True 时再乘 N_F/N，把"有效梯度被放大 N/N_F 倍"
                    # 这个混杂因子消掉，从而把"监督哪些样本"与"梯度多大"分开。
                    _l, _parts = self.freq_loss(logits, ep.query_peak, return_parts=True)
                    _ps = _parts['per_sample']
                    _keep = (ep.query_y != self.nc_index)
                    if bool(_keep.any()):
                        l_freq = (_ps[_keep].sum() / _ps.shape[0]) if self.mask_nc_rescale \
                            else _ps[_keep].mean()
                    else:
                        l_freq = _ps.mean()
                else:
                    l_freq = self.freq_loss(logits, ep.query_peak)
                stats['loss_freq'] = float(l_freq.detach())
                loss = loss + lam * l_freq
            else:
                stats['loss_freq'] = 0.0
            with torch.no_grad():
                acc = float((logits.argmax(1) == ep.query_y).float().mean())
            stats['train_acc'] = acc
            return loss, stats

        if self.method == 'ProtoNet':
            e_s = self._embed(model, ep.support_x)
            e_q = self._embed(model, ep.query_x)
            protos = self._prototypes(e_s, ep.support_y)         # (n_way, D)
            d2 = torch.cdist(e_q, protos) ** 2
            logits = -d2
            loss = F.cross_entropy(logits, ep.query_y)
            stats['loss_ce'] = float(loss.detach())
            stats['loss_freq'] = 0.0
            with torch.no_grad():
                stats['train_acc'] = float((logits.argmax(1) == ep.query_y).float().mean())
            return loss, stats

        if self.method == 'MatchingNet':
            e_s = self._embed(model, ep.support_x)
            e_q = self._embed(model, ep.query_x)
            cos = F.normalize(e_q, dim=1) @ F.normalize(e_s, dim=1).t()   # (nq, ns)
            attn = F.softmax(cos * self.rc.matching_scale, dim=1)
            onehot = F.one_hot(ep.support_y, num_classes=self.n_way).float()
            probs = attn @ onehot                                        # (nq, n_way)
            loss = F.nll_loss(torch.log(probs + 1e-12), ep.query_y)
            stats['loss_ce'] = float(loss.detach())
            stats['loss_freq'] = 0.0
            with torch.no_grad():
                stats['train_acc'] = float((probs.argmax(1) == ep.query_y).float().mean())
            return loss, stats

        # FineTune：训练阶段与 CNN-CE 完全一致，差别只在测试时微调
        logits = model(ep.query_x)
        loss = F.cross_entropy(logits, ep.query_y)
        stats['loss_ce'] = float(loss.detach())
        stats['loss_freq'] = 0.0
        with torch.no_grad():
            stats['train_acc'] = float((logits.argmax(1) == ep.query_y).float().mean())
        return loss, stats

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, model: BearingCNN, ep: Episode) -> torch.Tensor:
        """返回 query 的 logits/prob 型分数 ``(n_query, n_way)``。"""
        if self.method in ('CNN-CE', 'CNN-Freq'):
            return model(ep.query_x)
        if self.method == 'ProtoNet':
            e_s = self._embed(model, ep.support_x)
            e_q = self._embed(model, ep.query_x)
            return -(torch.cdist(e_q, self._prototypes(e_s, ep.support_y)) ** 2)
        if self.method == 'MatchingNet':
            e_s = self._embed(model, ep.support_x)
            e_q = self._embed(model, ep.query_x)
            cos = F.normalize(e_q, dim=1) @ F.normalize(e_s, dim=1).t()
            attn = F.softmax(cos * self.rc.matching_scale, dim=1)
            onehot = F.one_hot(ep.support_y, num_classes=self.n_way).float()
            probs = attn @ onehot
            return torch.log(probs + 1e-12)
        if self.method == 'FineTune':
            return self._finetune_predict(model, ep)
        raise ValueError(self.method)

    def _finetune_predict(self, model: BearingCNN, ep: Episode) -> torch.Tensor:
        """在支持集上微调一份**副本**后预测 query（不改动训练好的模型）。

        注意：本方法会被 ``predict`` 的 ``@torch.no_grad()`` 包住，
        而微调必须建图，故显式开 ``enable_grad``。
        """
        with torch.enable_grad():
            net = copy.deepcopy(model)
            net.train()
            opt = torch.optim.Adam(net.parameters(), lr=self.rc.finetune_lr)
            for _ in range(self.rc.finetune_steps):
                logits = net(ep.support_x)
                loss = F.cross_entropy(logits, ep.support_y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            return net(ep.query_x)

    def set_load_rpm(self, rpm: float):
        """跨工况时切到目标域标称转速。"""
        self.freq_loss.set_rpm(rpm)


# ----------------------------------------------------------------------
# 训练
# ----------------------------------------------------------------------
@dataclass
class TrainResult:
    method: str
    seed: int
    history: List[Dict[str, float]] = field(default_factory=list)
    best_val_f1: float = 0.0
    best_val_acc: float = 0.0
    best_epoch: int = -1
    best_state: Optional[dict] = None
    n_params: int = 0
    train_time_s: float = 0.0
    epoch_time_s: float = 0.0
    peak_gpu_mem_mb: float = 0.0
    flops_m: float = 0.0
    cfg_hash: str = ''

    def summary(self) -> dict:
        return {k: v for k, v in self.__dict__.items()
                if k not in ('history', 'best_state')}


def evaluate_episodes(model: BearingCNN, runner: MethodRunner, sampler: EpisodeSampler,
                      n_episodes: int, n_way: int,
                      collect: bool = False, device: str = 'cpu') -> dict:
    """在固定 episode 序列上评估，返回 acc / macro-F1 与（可选）混淆矩阵、混淆样本。"""
    was_training = model.training
    model.eval()
    accs, f1s = [], []
    cm = np.zeros((n_way, n_way), dtype=np.int64)
    cm_sup = np.zeros((n_way, n_way), dtype=np.int64)
    collect_data = {'emb': [], 'y': [], 'peak': [], 'logits': []} if collect else None
    n_sup_samples = len(sampler.pool) and sampler.n_shot * n_way

    with torch.no_grad():
        for ep in sampler.episodes(n_episodes):
            ep = ep.to(device)
            logits = runner.predict(model, ep)
            pred = logits.argmax(dim=1)
            yt = ep.query_y
            accs.append(float((pred == yt).float().mean()))
            f1s.append(macro_f1(yt.cpu().numpy(), pred.cpu().numpy(), n_way))
            cm += confusion_matrix(yt.cpu().numpy(), pred.cpu().numpy(), n_way)
            # 支持集上也不应该失衡，顺带统计（用于诊断"模型没学会"还是"泛化失败"）
            sup_logits = model(ep.support_x)
            cm_sup += confusion_matrix(ep.support_y.cpu().numpy(),
                                       sup_logits.argmax(1).cpu().numpy(), n_way)
            if collect_data is not None:
                collect_data['emb'].append(model.embed(ep.query_x).cpu().numpy())
                collect_data['y'].append(yt.cpu().numpy())
                collect_data['peak'].append(ep.query_peak.cpu().numpy())
                collect_data['logits'].append(logits.cpu().numpy())
    if was_training:
        model.train()

    out = {
        'acc': float(np.mean(accs)), 'acc_std': float(np.std(accs)),
        'f1': float(np.mean(f1s)), 'f1_std': float(np.std(f1s)),
        'acc_per_episode': accs, 'f1_per_episode': f1s,
        'confusion_matrix': cm.tolist(), 'confusion_matrix_support': cm_sup.tolist(),
    }
    if collect_data is not None:
        out['collect'] = {k: np.concatenate(v, axis=0) for k, v in collect_data.items()}
    return out


def fit(cfg: dict, train_pool: WindowPool, val_pool: WindowPool, method: str,
        rc: RunConfig, out_dir: Optional[str] = None, verbose: bool = True) -> TrainResult:
    """训练一个方法，返回结果（含最优权重）。"""
    device = resolve_device(rc.device)
    setup_seed(rc.seed, verbose=False)

    model = BearingCNN.from_cfg(cfg).to(device)
    n_params = count_parameters(model)
    flops = estimate_flops(model, int(cfg['data']['window']))
    # 注意：必须 .to(device)。H/f0/valid 是注册 buffer，漏掉这步会让它们在 CPU，
    # 与已上 GPU 的 episode.query_peak 相减时报 "found at least two devices"。
    freq_loss = FrequencySoftConstraintLoss.from_cfg(cfg).to(device)
    runner = MethodRunner(method, freq_loss, cfg, rc)
    sched = build_lambda_scheduler(cfg['loss'])

    opt = torch.optim.AdamW(model.parameters(), lr=rc.lr, weight_decay=rc.weight_decay)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, rc.epochs), eta_min=float(cfg['train'].get('cosine_eta_min', 0.0)))

    train_sampler = EpisodeSampler.from_cfg(train_pool, cfg, seed=derive_seed(rc.seed, 'train'),
                                            device=device, name='train')
    # 验证用固定种子 -> 每次评估看的是同一批 episode，epoch 之间可比
    val_sampler = EpisodeSampler.from_cfg(val_pool, cfg, seed=derive_seed(rc.seed, 'val'),
                                          device=device, name='val')

    res = TrainResult(method=method, seed=rc.seed, n_params=n_params,
                      flops_m=flops['flops_m'], cfg_hash=lg.cfg_hash(cfg))
    t_start = time.time()
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(rc.epochs):
        lam = sched.value(epoch)
        model.train()
        ep_loss, ep_ce, ep_freq, ep_acc, n = 0.0, 0.0, 0.0, 0.0, 0
        te = time.time()
        for _ in range(rc.episodes_per_epoch):
            ep = train_sampler.sample().to(device)
            loss, st = runner.train_step(model, ep, lam)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if rc.grad_clip and rc.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), rc.grad_clip)
            opt.step()
            ep_loss += float(loss.detach()); ep_ce += st.get('loss_ce', 0.0)
            ep_freq += st.get('loss_freq', 0.0); ep_acc += st.get('train_acc', 0.0); n += 1
        lr_sched.step()
        row = {'epoch': epoch, 'lambda': lam, 'lr': opt.param_groups[0]['lr'],
               'loss_total': ep_loss / n, 'loss_ce': ep_ce / n, 'loss_freq': ep_freq / n,
               'train_acc': ep_acc / n, 'epoch_time_s': time.time() - te}
        row['val_acc'] = row['val_f1'] = float('nan')
        if (epoch + 1) % rc.val_every == 0 or epoch == rc.epochs - 1:
            v = evaluate_episodes(model, runner, val_sampler, rc.val_episodes, rc.n_way,
                                  collect=False, device=device)
            row['val_acc'], row['val_f1'] = v['acc'], v['f1']
            if v['f1'] > res.best_val_f1 - 1e-12:
                res.best_val_f1, res.best_val_acc, res.best_epoch = v['f1'], v['acc'], epoch
                res.best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        res.history.append(row)
        if verbose and (epoch % max(1, rc.epochs // 10) == 0 or epoch == rc.epochs - 1):
            print('  [%s seed=%d] epoch %3d/%d lam=%.2f loss=%.4f ce=%.4f freq=%.4f '
                  'train_acc=%.3f val_acc=%.3f val_f1=%.3f (%.1fs)'
                  % (method, rc.seed, epoch + 1, rc.epochs, lam, row['loss_total'],
                     row['loss_ce'], row['loss_freq'], row['train_acc'],
                     row['val_acc'], row['val_f1'], row['epoch_time_s']))

    res.train_time_s = time.time() - t_start
    res.epoch_time_s = res.train_time_s / max(1, rc.epochs)
    if device == 'cuda':
        res.peak_gpu_mem_mb = torch.cuda.max_memory_allocated() / 1e6

    # 回滚到最优权重（按验证集 macro-F1）
    if res.best_state is not None:
        model.load_state_dict(res.best_state)

    if out_dir:
        lg.ensure_dirs(out_dir)
        lg.write_csv(os.path.join(out_dir, 'history.csv'), res.history)
        lg.write_json(os.path.join(out_dir, 'summary.json'), res.summary())
        lg.write_json(os.path.join(out_dir, 'scheduler.json'), sched.describe())
        torch.save({'state_dict': model.state_dict(), 'cfg': model.cfg,
                    'method': method, 'seed': rc.seed, 'n_params': n_params},
                   os.path.join(out_dir, 'best.pt'))
        lg.write_env_snapshot(out_dir)
    res._model = model                                                  # 便于同进程内复用
    return res


def load_model_from_ckpt(path: str, cfg: dict, device: str = 'cpu') -> BearingCNN:
    ck = torch.load(path, map_location=device, weights_only=False)
    m = BearingCNN(**{k: v for k, v in ck['cfg'].items()}).to(device)
    m.load_state_dict(ck['state_dict'])
    m.eval()
    return m


# ----------------------------------------------------------------------
# 配置装载
# ----------------------------------------------------------------------
def load_cfg(path: Optional[str] = None) -> dict:
    import yaml
    if path is None:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'configs', 'default.yaml')
    with open(path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg


def absolutize_paths(cfg: dict, root: str) -> dict:
    for k, v in list(cfg['paths'].items()):
        cfg['paths'][k] = v if os.path.isabs(v) else os.path.join(root, v)
    for k in ('raw_dir', 'processed_dir'):
        v = cfg['data'][k]
        cfg['data'][k] = v if os.path.isabs(v) else os.path.join(root, v)
    return cfg


if __name__ == '__main__':
    # 冒烟测试：用合成池跑通全部 5 个方法的 3 个 epoch
    from data.cwru_loader import WindowPool

    cfg = load_cfg()
    cfg['freq']['shaft_rpm'] = {int(k): float(v) for k, v in cfg['freq']['shaft_rpm'].items()}
    rng = np.random.default_rng(0)
    n_per_c, L, K = 60, int(cfg['data']['window']), int(cfg['freq']['n_peaks'])
    pools = {}
    for name in ('train', 'val'):
        X = rng.standard_normal((n_per_c * 4, L)).astype(np.float32)
        y = np.repeat(np.arange(4), n_per_c).astype(np.int64)
        pk = rng.uniform(50, 500, size=(n_per_c * 4, K)).astype(np.float32)
        pools[name] = WindowPool(X=X, y=y, load=np.zeros_like(y), peak_f=pk)

    device = resolve_device('auto')
    print('device =', device)
    rc = RunConfig(epochs=3, episodes_per_epoch=10, val_every=1, val_episodes=5,
                   device=device, finetune_steps=3, progress=False)
    for m in METHODS:
        r = fit(cfg, pools['train'], pools['val'], m, rc, out_dir=None, verbose=False)
        print('%-12s params=%d  best_val_acc=%.3f best_val_f1=%.3f  %.2fs/epoch'
              % (m, r.n_params, r.best_val_acc, r.best_val_f1, r.epoch_time_s))
    print('\n[OK] 5 个方法全部跑通')
