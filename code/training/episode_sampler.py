# -*- coding: utf-8 -*-
"""Few-shot episode 采样器（4-way 5-shot）。

协议（与项目文档逐字一致）
--------------------------
- ``n_way = 4``：NC / IR007 / OR007 / B007 四类全取（CWRU 这小样本协议里 way 数=类别数）
- ``n_shot = 5``：每类 5 个 support  → support = 4 × 5 = **20**
- ``n_query_per_class = 15``：每类 15 个 query → query = 4 × 15 = **60**

采样细节
--------
- **有放回采样**：CWRU 每类可用窗口只有几十到几百个，无放回会在多 episode 训练中迅速耗尽，
  这也是 few-shot 文献的通行做法；
- 一个 episode 内同一类的 support/query **不强制互斥**（有放回必然可能重复），
  但 support 与 query 各自独立采样；
- 用 ``torch.Generator`` + 派生种子，保证可复现；
- ``classes`` 可指定用哪些类做 episode（4-way 时即全取），
  便于做"3-way/4-way"的稳健性检查。

性能说明
--------
窗口池很小（数百~数千条），所以不做 DataLoader，而是把整池张量**常驻显存**，
每个 episode 只采样索引 → 单 episode 的 host-device 通信几乎为零。
实测这一步比 ``DataLoader`` 快一个数量级，且完全确定性。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

try:
    from ..data.cwru_loader import WindowPool
except (ImportError, ValueError):                                     # pragma: no cover
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.cwru_loader import WindowPool


@dataclass
class Episode:
    support_x: torch.Tensor        # (n_way*n_shot, L)
    support_y: torch.Tensor        # (n_way*n_shot,)  已在 episode 内重映射到 0..n_way-1
    support_peak: torch.Tensor     # (n_way*n_shot, K)
    query_x: torch.Tensor          # (n_way*n_query, L)
    query_y: torch.Tensor          # (n_way*n_query,)
    query_peak: torch.Tensor       # (n_way*n_query, K)
    classes: List[int]             # 本 episode 用到的原始类别下标（升序）
    load: int                      # 本 episode 的工况 (hp)，用于选 f_theory

    def to(self, device) -> 'Episode':
        return Episode(
            support_x=self.support_x.to(device), support_y=self.support_y.to(device),
            support_peak=self.support_peak.to(device),
            query_x=self.query_x.to(device), query_y=self.query_y.to(device),
            query_peak=self.query_peak.to(device),
            classes=self.classes, load=self.load)

    @property
    def n_support(self) -> int:
        return int(self.support_x.shape[0])

    @property
    def n_query(self) -> int:
        return int(self.query_x.shape[0])

    def summary(self) -> dict:
        return dict(n_support=self.n_support, n_query=self.n_query,
                    classes=self.classes, load=self.load,
                    support_per_class=(self.n_support // max(1, len(self.classes))),
                    query_per_class=(self.n_query // max(1, len(self.classes))))


class EpisodeSampler:
    """从 ``WindowPool`` 上采样 episode。"""

    def __init__(self, pool: WindowPool, n_way: int = 4, n_shot: int = 5,
                 n_query_per_class: int = 15, seed: int = 0, device='cpu',
                 preload: bool = True, name: str = 'pool',
                 disjoint: bool = False):
        self.pool = pool
        self.n_way = int(n_way)
        self.n_shot = int(n_shot)
        self.n_query = int(n_query_per_class)
        self.name = name
        # 支持集与查询集是否互斥（True 时合并抽取后切分；默认 False = 历史行为）
        self.disjoint = bool(disjoint)
        self.rng = np.random.default_rng(seed)
        self._gen = torch.Generator(device='cpu').manual_seed(int(seed) % (2 ** 31 - 1))

        self.y = np.asarray(pool.y)
        n_classes = int(self.y.max()) + 1 if self.y.size else 0
        if self.n_way > n_classes:
            raise ValueError('n_way=%d 超过池中类别数 %d' % (self.n_way, n_classes))
        self.n_classes_total = n_classes
        # 每个类别的样本下标
        self.class_idx: Dict[int, np.ndarray] = {
            c: np.flatnonzero(self.y == c) for c in range(n_classes)}
        sizes = {c: int(v.size) for c, v in self.class_idx.items()}
        need = self.n_shot + self.n_query
        too_small = {c: s for c, s in sizes.items() if s < 1}
        self.class_sizes = sizes
        self.min_needed_per_class = need

        if preload:
            self.X = torch.from_numpy(np.ascontiguousarray(pool.X)).to(device)
            self.PK = torch.from_numpy(np.ascontiguousarray(pool.peak_f)).to(device)
        else:                                                            # pragma: no cover
            self.X = torch.from_numpy(np.ascontiguousarray(pool.X))
            self.PK = torch.from_numpy(np.ascontiguousarray(pool.peak_f))
        self.device = device
        # 各工况：同池内负载可能不唯一（跨工况池是按目标负载单独建的，故通常唯一）
        loads = np.unique(pool.load)
        self.loads = loads.tolist()
        self._too_small = too_small

    # ------------------------------------------------------------------
    def sample(self, classes: Optional[Sequence[int]] = None) -> Episode:
        pool_classes = (list(classes) if classes is not None
                        else list(range(self.n_classes_total)))
        if len(pool_classes) > self.n_way:
            pool_classes = list(self.rng.choice(pool_classes, size=self.n_way, replace=False))
        chosen = sorted(int(c) for c in pool_classes)
        for c in chosen:
            if self.class_idx[c].size == 0:
                raise ValueError('类别 %d 在池 %s 中无样本' % (c, self.name))

        sup_i, sup_y, qry_i, qry_y = [], [], [], []
        for new_lab, c in enumerate(chosen):
            idx = self.class_idx[c]
            # 有放回采样（每类可用窗口有限，无放回会迅速耗尽）
            # disjoint=True：支持集与查询集互斥 —— 合并后无放回抽取再切分。
            # 用于"支持/查询重叠"的敏感性对照，默认关闭以保证与历史结果可比。
            if self.disjoint and idx.size >= self.n_shot + self.n_query:
                _pick = self.rng.choice(idx, size=self.n_shot + self.n_query, replace=False)
                s, q = _pick[:self.n_shot], _pick[self.n_shot:]
            else:
                s = self.rng.choice(idx, size=self.n_shot, replace=True)
                q = self.rng.choice(idx, size=self.n_query, replace=True)
            sup_i.append(s); sup_y.append(np.full(self.n_shot, new_lab))
            qry_i.append(q); qry_y.append(np.full(self.n_query, new_lab))
        sup_i = np.concatenate(sup_i); qry_i = np.concatenate(qry_i)
        sup_y = np.concatenate(sup_y).astype(np.int64)
        qry_y = np.concatenate(qry_y).astype(np.int64)

        # 用 tensor 索引，避免 python 逐条取数
        si = torch.from_numpy(sup_i).long()
        qi = torch.from_numpy(qry_i).long()
        load = int(self.loads[0]) if len(self.loads) == 1 else int(self.loads[0])
        return Episode(
            support_x=self.X[si], support_y=torch.from_numpy(sup_y).long(),
            support_peak=self.PK[si],
            query_x=self.X[qi], query_y=torch.from_numpy(qry_y).long(),
            query_peak=self.PK[qi],
            classes=chosen, load=load)

    # ------------------------------------------------------------------
    def episodes(self, n: int, classes: Optional[Sequence[int]] = None):
        for _ in range(int(n)):
            yield self.sample(classes)

    def class_distribution(self) -> Dict[int, int]:
        return dict(self.class_sizes)

    def describe(self) -> dict:
        return dict(pool=self.name, n=int(len(self.pool)),
                    n_way=self.n_way, n_shot=self.n_shot,
                    n_query_per_class=self.n_query,
                    support_total=self.n_way * self.n_shot,
                    query_total=self.n_way * self.n_query,
                    class_sizes=self.class_sizes, loads=self.loads)

    @classmethod
    def from_cfg(cls, pool: WindowPool, cfg: dict, seed: int = 0, device='cpu',
                 name: str = 'pool') -> 'EpisodeSampler':
        fs = cfg['fewshot']
        return cls(pool, int(fs['n_way']), int(fs['n_shot']),
                   int(fs['n_query_per_class']), seed=seed, device=device, name=name,
                   disjoint=bool(fs.get('disjoint_support_query', False)))


# ----------------------------------------------------------------------
if __name__ == '__main__':
    from data.cwru_loader import WindowPool

    # 构造一个合成的池做冒烟测试
    rng = np.random.default_rng(0)
    n_per_c, L, K = 37, 2048, 5
    X = rng.standard_normal((n_per_c * 4, L)).astype(np.float32)
    y = np.repeat(np.arange(4), n_per_c).astype(np.int64)
    pk = rng.uniform(50, 500, size=(n_per_c * 4, K)).astype(np.float32)
    pool = WindowPool(X=X, y=y, load=np.zeros_like(y), peak_f=pk)

    s = EpisodeSampler(pool, n_way=4, n_shot=5, n_query_per_class=15, seed=42, name='demo')
    print('=== 采样器描述 ===')
    for k, v in s.describe().items():
        print('  %-22s %s' % (k, v))

    ep = s.sample()
    print('\n=== 一个 episode ===')
    for k, v in ep.summary().items():
        print('  %-22s %s' % (k, v))
    print('  support_x %s  query_x %s' % (tuple(ep.support_x.shape), tuple(ep.query_x.shape)))
    print('  support_y 分布 %s' % torch.bincount(ep.support_y, minlength=4).tolist())
    print('  query_y   分布 %s' % torch.bincount(ep.query_y, minlength=4).tolist())
    assert ep.support_x.shape == (20, 2048)
    assert ep.query_x.shape == (60, 2048)
    assert ep.support_peak.shape == (20, 5)
    assert ep.query_peak.shape == (60, 5)

    # 可复现性
    s2 = EpisodeSampler(pool, 4, 5, 15, seed=42, name='demo')
    ep_b = s2.sample()
    same = torch.equal(ep.support_x, ep_b.support_x) and torch.equal(ep.query_x, ep_b.query_x)
    print('\n同种子复现: %s' % ('PASS' if same else 'FAIL'))
    assert same

    s3 = EpisodeSampler(pool, 4, 5, 15, seed=43, name='demo')
    diff = not torch.equal(ep.support_x, s3.sample().support_x)
    print('异种子不同: %s' % ('PASS' if diff else 'FAIL'))
    assert diff

    # 采样覆盖度：多 episode 后每个类别都被用到
    seen = set()
    for e in s.episodes(30):
        seen.add(tuple(e.classes))
    print('\n30 个 episode 的类别组合数 = %d -> %s'
          % (len(seen), 'PASS' if len(seen) == 1 else 'FAIL(4-way 应恒为全部 4 类)'))
