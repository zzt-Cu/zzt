# -*- coding: utf-8 -*-
"""CWRU 12kHz 驱动端数据加载、切窗、划分与缓存。

本模块刻意绕开了 CWRU 数据集三个"不报错但让结论悄悄错掉"的坑（详见下）：

坑① 同一 .mat 里有多个 ``*_DE_time`` 变量
    实测 ``99.mat`` 内部**同时**含 ``X098_DE_time``(1HP) 与 ``X099_DE_time``(2HP)。
    若按"第一个/最后一个以 _DE_time 结尾的变量"取值，2HP 样本会被悄悄换成 1HP，
    跨工况实验直接退化成"源域=目标域"且不报错。
    ⇒ 本模块一律用 ``de_var_name(code) = X{code:03d}_DE_time`` **显式按官方编号取值**，
      并把"文件里出现了几个 DE 变量、分别叫什么"记录进 manifest 供人工复核。

坑② 文件长度差数倍 ⇒ 类别失衡
    正常基线记录远长于故障记录，不加限制地滑窗会让正常类窗口数成倍压过其他类，
    accuracy 被刷高。
    ⇒ 逐文件等间隔截取上限 ``max_windows_per_file``，并把**每个文件的窗口数与被截断情况**
      打印出来；指标使用 macro-F1。

坑③ ``Path.stem`` 会在带小数点的规范化文件名上截断（``..._0.007in_...`` → ``..._0``）
    ⇒ 提供 ``bare_name()``：只从右侧显式剥 ``.mat``，且对 ``path.name`` 调用。

划分策略（杜绝信息泄漏）
------------------------
每条记录按时间切成互斥的连续三段 60% / 20% / 20% = train / val / test，
再在各自时间块内滑窗。因为窗口不跨块边界，**任何 val/test 窗口都不会与 train 窗口重叠**。
跨工况实验中，目标域（1/2/3 hp）**整条记录**都从未参与训练。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:                                    # 允许直接以脚本方式导入
    from . import bearing_freq as bf
    from .envelope_spectrum import EnvelopeConfig, peak_frequencies
except ImportError:                     # pragma: no cover
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data import bearing_freq as bf
    from data.envelope_spectrum import EnvelopeConfig, peak_frequencies


# ----------------------------------------------------------------------
# 官方编号 <-> 类别/尺寸/负载
# ----------------------------------------------------------------------
# 默认类别顺序（4 类主实验）。**标签 = 列表下标**，所以换类别集合就是换标签空间：
# 窗口池缓存指纹与 n_way 必须跟着变（见 cache_path 与 evaluate.py 的 --classes）。
CLASS_ORDER: List[str] = ['NC', 'IR007', 'OR007', 'B007']
LABEL_MAP: Dict[str, int] = {c: i for i, c in enumerate(CLASS_ORDER)}


def label_map(classes: Optional[Sequence[str]] = None) -> Dict[str, int]:
    """类别列表 -> 标签映射；``classes=None`` 时用默认 ``CLASS_ORDER``。

    ⚠️ 调用方必须传入**本次实验实际使用的类别列表**，不要图省事直接用模块级 ``LABEL_MAP``：
    改成 3 类（如 NC/IR007/OR007）后若仍用默认映射，B007 的下标 3 会落到越界标签上，
    窗口池缓存也会被错误复用 —— 这类错**不报错**，只让结果悄悄对不上。
    """
    seq = list(classes) if classes is not None else list(CLASS_ORDER)
    return {c: i for i, c in enumerate(seq)}

# 缺陷尺寸（inch）；NC 无缺陷
DEFECT_SIZE_IN: Dict[str, Optional[float]] = {
    'NC': None,
    'IR007': 0.007, 'OR007': 0.007, 'B007': 0.007,
    'IR014': 0.014, 'OR014': 0.014, 'B014': 0.014,
    'IR021': 0.021, 'OR021': 0.021, 'B021': 0.021,
}
# 外圈故障位置：CWRU 12kHz 驱动端 OR 统一取 @6 点钟（承压区中心），保证跨尺寸可比
FAULT_POSITION: Dict[str, Optional[str]] = {
    'NC': None,
    'IR007': 'inner_race', 'OR007': 'outer_race@6', 'B007': 'ball',
    'IR014': 'inner_race', 'OR014': 'outer_race@6', 'B014': 'ball',
    'IR021': 'inner_race', 'OR021': 'outer_race@6', 'B021': 'ball',
}

# 官方编号表：类别 -> {负载 hp: 官方文件编号}
# 编号来源：https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data
# （2026-09-22 逐条核实，非凭记忆 —— 例如 OR021@6 的 0hp 是 234 而不是 235）
FILE_CODES: Dict[str, Dict[int, int]] = {
    # ---- 0.007"：主实验 4 类 ----
    'NC':    {0: 97,  1: 98,  2: 99,  3: 100},
    'IR007': {0: 105, 1: 106, 2: 107, 3: 108},
    'B007':  {0: 118, 1: 119, 2: 120, 3: 121},
    'OR007': {0: 130, 1: 131, 2: 132, 3: 133},
    # ---- 0.014"：severity shift（OR014 仅 @6:00 有数据）----
    'IR014': {0: 169, 1: 170, 2: 171, 3: 172},
    'B014':  {0: 185, 1: 186, 2: 187, 3: 188},
    'OR014': {0: 197, 1: 198, 2: 199, 3: 200},
    # ---- 0.021"：severity shift（取 @6:00，与 0.007/0.014 同位置）----
    'IR021': {0: 209, 1: 210, 2: 211, 3: 212},
    'B021':  {0: 222, 1: 223, 2: 224, 3: 225},
    'OR021': {0: 234, 1: 235, 2: 236, 3: 237},
}

# 各负载标称转速 (r/min)。⚠️ 实测转速与标称值有 ±10 r/min 级别偏差。
NOMINAL_RPM: Dict[int, float] = {0: 1797.0, 1: 1772.0, 2: 1750.0, 3: 1730.0}

DE_KEY_RE = 'DE_time'


def de_var_name(code: int) -> str:
    """官方编号 -> DE 变量名。``de_var_name(105) -> 'X105_DE_time'``"""
    return 'X%03d_DE_time' % int(code)


def bare_name(value) -> str:
    """从路径/文件名安全地剥掉 ``.mat`` 后缀。

    **不要用 ``Path(...).stem``**：规范化文件名里含 ``0.007`` 这样的点，
    ``.stem`` 会从最后一个点开始剥，把 ``..._0.007in_...`` 截成 ``..._0``，
    导致解析静默失败。
    """
    name = Path(str(value)).name               # 注意：.name 而非 .stem
    if name.lower().endswith('.mat'):
        return name[:-4]
    return name


# ----------------------------------------------------------------------
# 原始信号读取
# ----------------------------------------------------------------------
@dataclass
class RawRecord:
    cls: str
    load_hp: int
    code: int
    path: str
    de_key: str
    n_samples: int
    all_de_keys: List[str] = field(default_factory=list)
    extra_de_keys: List[str] = field(default_factory=list)
    rms: float = float('nan')
    peak: float = float('nan')
    kurtosis: float = float('nan')
    sha256_16: str = ''
    warnings: List[str] = field(default_factory=list)


def _mat_de_keys(mat: dict) -> List[str]:
    return sorted(k for k in mat.keys() if k.endswith(DE_KEY_RE))


def load_de_signal(code: int, raw_dir: str, fs: Optional[int] = None,
                   verbose: bool = False) -> Tuple[np.ndarray, RawRecord]:
    """按官方编号读取 DE 信号（float64, 1-D）。

    Raises
    ------
    FileNotFoundError : 文件不存在
    KeyError          : 文件内缺少 ``X{code:03d}_DE_time``
    """
    from scipy.io import loadmat
    import hashlib

    path = os.path.join(raw_dir, '%d.mat' % int(code))
    if not os.path.exists(path):
        raise FileNotFoundError('缺少数据文件: %s' % path)

    mat = loadmat(path)
    want = de_var_name(code)
    all_keys = _mat_de_keys(mat)
    if want not in mat:
        raise KeyError('文件 %s 内找不到变量 %s；实际 DE 变量=%s'
                       % (os.path.basename(path), want, all_keys))

    sig = np.asarray(mat[want]).squeeze().astype(np.float64)
    if sig.ndim != 1:
        raise ValueError('%s 形状异常: %s' % (want, np.asarray(mat[want]).shape))

    extra = [k for k in all_keys if k != want]
    cls, load_hp = code_to_class_load(code)
    rec = RawRecord(
        cls=cls, load_hp=load_hp, code=int(code), path=path, de_key=want,
        n_samples=int(sig.size), all_de_keys=all_keys, extra_de_keys=extra,
        rms=float(np.sqrt((sig ** 2).mean())),
        peak=float(np.abs(sig).max()),
        kurtosis=float(((sig - sig.mean()) ** 4).mean() / (sig.var() ** 2 + 1e-30)),
        sha256_16=hashlib.sha256(open(path, 'rb').read()).hexdigest()[:16],
    )
    if extra:
        rec.warnings.append(
            '同文件内含额外 DE 变量 %s（已知 99.mat 会混入 X098_DE_time）；'
            '本模块按官方编号 %s 取值，未受影响' % (extra, want))
    if verbose:
        print('  [%s] %s <- %s  n=%d  rms=%.4f  kurt=%.2f%s'
              % (cls, os.path.basename(path), want, rec.n_samples, rec.rms,
                 rec.kurtosis, ('  WARN: ' + '; '.join(rec.warnings)) if rec.warnings else ''))
    return sig, rec


def code_to_class_load(code: int) -> Tuple[str, int]:
    for cls, m in FILE_CODES.items():
        for hp, c in m.items():
            if c == int(code):
                return cls, int(hp)
    raise KeyError('未知官方编号: %s' % code)


# ----------------------------------------------------------------------
# RAW_MANIFEST.csv 与交叉核对
# ----------------------------------------------------------------------
def build_raw_manifest(cfg: dict, verbose: bool = True) -> List[RawRecord]:
    """遍历配置中全部编号，读取并落 ``data/raw/RAW_MANIFEST.csv``。

    返回逐条记录。抽样核对字段：
    ``n_expected / n_found / missing / mismatch / extra_de_keys / ok``
    —— 找不到文件时也返回结构化结果，不让调用方 KeyError（便于巡查脚本判断）。
    """
    raw_dir = cfg['data']['raw_dir']
    fs = int(cfg['data']['fs'])
    codes = [(cls, hp, c) for cls, m in FILE_CODES.items() for hp, c in sorted(m.items())]
    recs: List[RawRecord] = []
    missing: List[int] = []
    for cls, hp, code in codes:
        try:
            _, rec = load_de_signal(code, raw_dir, fs, verbose=verbose)
            # 只有 cfg 中列出的类别才纳入（本配置为全部）
            if cls not in cfg['data']['classes']:
                rec.warnings.append('类别 %s 不在 cfg.data.classes 中' % cls)
            recs.append(rec)
        except (FileNotFoundError, KeyError) as e:
            missing.append(code)
            recs.append(RawRecord(cls=cls, load_hp=hp, code=code, path='',
                                  de_key=de_var_name(code), n_samples=0,
                                  warnings=['READ_FAIL: %s' % e]))

    mism = [r.code for r in recs
            if r.path and os.path.basename(r.path) != '%d.mat' % r.code]
    cross = {
        'n_expected': len(codes),
        'n_found': sum(1 for r in recs if r.path),
        'missing': missing,
        'mismatch': mism,
        'files_with_extra_de_keys': [(r.code, r.extra_de_keys) for r in recs if r.extra_de_keys],
        'ok': (not missing) and (not mism) and len(recs) == len(codes),
    }

    # 落 CSV
    if recs:
        import csv
        csv_path = os.path.join(raw_dir, 'RAW_MANIFEST.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['code', 'file', 'class', 'fault_position', 'defect_size_in',
                        'load_hp', 'nominal_rpm', 'de_var', 'n_samples', 'fs',
                        'rms', 'peak', 'kurtosis', 'sha256_16',
                        'all_de_keys', 'extra_de_keys', 'warnings'])
            for r in recs:
                w.writerow([r.code, '%d.mat' % r.code, r.cls,
                            FAULT_POSITION.get(r.cls), DEFECT_SIZE_IN.get(r.cls),
                            r.load_hp, NOMINAL_RPM.get(r.load_hp), r.de_key,
                            r.n_samples, fs, '%.4f' % r.rms if r.path else '',
                            '%.3f' % r.peak if r.path else '',
                            '%.3f' % r.kurtosis if r.path else '',
                            r.sha256_16, ';'.join(r.all_de_keys),
                            ';'.join(r.extra_de_keys), ';'.join(r.warnings)])
    if verbose:
        print('=== RAW_MANIFEST 交叉核对 ===')
        for k in ['n_expected', 'n_found', 'missing', 'mismatch', 'ok']:
            print('  %-26s %s' % (k, cross[k]))
        print('  files_with_extra_de_keys  %s' % cross['files_with_extra_de_keys'])
    return recs


# ----------------------------------------------------------------------
# 切窗 / 截断 / 标准化
# ----------------------------------------------------------------------
def segment(sig: np.ndarray, window: int, stride: int) -> np.ndarray:
    """滑窗切分，返回 ``(N, window)``（视图拷贝，float32）。"""
    n = sig.size
    if n < window:
        return np.zeros((0, window), dtype=np.float32)
    starts = np.arange(0, n - window + 1, stride)
    idx = starts[:, None] + np.arange(window)[None, :]
    return np.ascontiguousarray(sig[idx]).astype(np.float32)


def cap_windows(X: np.ndarray, max_n: int) -> Tuple[np.ndarray, np.ndarray]:
    """等间隔截取至多 ``max_n`` 个窗口，返回 ``(X', kept_idx)``。

    等间隔（而非取前 N 个）保证截断后的窗口在整段记录上均匀分布，
    不会只集中在记录开头。
    """
    if max_n <= 0 or X.shape[0] <= max_n:
        return X, np.arange(X.shape[0])
    kept = np.linspace(0, X.shape[0] - 1, max_n).round().astype(int)
    kept = np.unique(kept)
    return X[kept], kept


def zscore_windows(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """逐窗口零均值/单位方差。**统计量只来自该窗口自身**，不跨样本，
    因此不引入任何训练/测试集之间的信息泄漏。

    ⚠️ 不要改成"用全局均值方差"——那会用测试集统计量，属于泄漏
    （有 tests/test_loader.py 里的测试守住）。
    """
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return ((X - mu) / (sd + eps)).astype(np.float32)


def split_blocks(n: int, fractions: Sequence[float] = (0.6, 0.2, 0.2)
                 ) -> List[Tuple[int, int]]:
    """把 [0, n) 切成互斥且完备的连续块。"""
    fr = np.asarray(fractions, dtype=float)
    if abs(fr.sum() - 1.0) > 1e-6:
        raise ValueError('fractions 之和必须为 1，当前=%s' % fr.sum())
    edges = np.round(np.cumsum(np.concatenate([[0.0], fr])) * n).astype(int)
    edges[0], edges[-1] = 0, n
    return [(int(edges[i]), int(edges[i + 1])) for i in range(len(fr))]


# ----------------------------------------------------------------------
# 噪声（时域加噪，供鲁棒性实验；必须在波形上加，不能在包络谱上加）
# ----------------------------------------------------------------------
def add_noise_at_snr(X: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """按目标 SNR (dB) 加高斯白噪声。``SNR = 10*log10(P_signal / P_noise)``。"""
    P_s = (X.astype(np.float64) ** 2).mean(axis=1, keepdims=True)
    P_n = P_s / (10.0 ** (snr_db / 10.0))
    noise = rng.standard_normal(X.shape) * np.sqrt(P_n)
    return (X + noise).astype(np.float32)


def measure_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    """从加噪前后波形反算真实 SNR (dB)，用于自检。"""
    n = (noisy.astype(np.float64) - clean.astype(np.float64))
    return float(10.0 * np.log10((clean.astype(np.float64) ** 2).mean()
                                 / ((n ** 2).mean() + 1e-30)))


def add_noise_to_pools(pools: Dict[str, 'WindowPool'], snr_db: float, seed: int,
                       ecfg: 'EnvelopeConfig', verbose: bool = True
                       ) -> Dict[str, 'WindowPool']:
    """对全部窗口池做**时域**加噪，并对加噪波形**重新做包络特征化**。

    为什么不能在包络谱上加噪
    包络谱是非负量，叠加高斯噪声会造出物理上不存在的负谱线，结果没有意义。
    正确链路：干净波形 -> 加噪波形 -> 用同一套带通/峰值参数重新特征化 -> 评估同一份权重。

    同时会反算真实 SNR 并断言与请求值偏差 < 1 dB（每个池都查）。
    """
    rng = np.random.default_rng(derive_seed_local(seed, 'noise', snr_db))
    out: Dict[str, 'WindowPool'] = {}
    for name, wp in pools.items():
        Xn = add_noise_at_snr(wp.X, snr_db, rng)
        got = measure_snr(wp.X, Xn)
        if abs(got - snr_db) > 1.0:
            raise AssertionError('池 %s 的反算 SNR=%.2f dB 与请求 %.2f dB 偏差过大'
                                 % (name, got, snr_db))
        pk = peak_frequencies_batch_safe(Xn, ecfg)
        out[name] = WindowPool(X=Xn, y=wp.y, load=wp.load, peak_f=pk,
                               n_per_class=dict(wp.n_per_class))
        if verbose:
            print('  [noise] %-9s 请求 %.1f dB -> 反算 %.2f dB  (peak 已重算)'
                  % (name, snr_db, got))
    return out


def derive_seed_local(seed: int, *tags) -> int:
    import hashlib
    payload = '%d|%s' % (int(seed), '|'.join(str(t) for t in tags))
    return int(hashlib.sha256(payload.encode()).hexdigest()[:8], 16) % (2 ** 31 - 1)


# ----------------------------------------------------------------------
# 数据集构建
# ----------------------------------------------------------------------
@dataclass
class WindowPool:
    X: np.ndarray          # (N, L) float32, 已 z-score
    y: np.ndarray          # (N,)   int64
    load: np.ndarray       # (N,)   int64  (hp)
    peak_f: np.ndarray     # (N, K) float32, 谱峰频率（Hz），与负载无关
    n_per_class: Dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def summary(self) -> dict:
        return dict(n=int(self.X.shape[0]), window=int(self.X.shape[1]),
                    n_peaks=int(self.peak_f.shape[1]),
                    n_per_class=self.n_per_class,
                    n_distinct_peak_rows=int(np.unique(self.peak_f, axis=0).shape[0]))


def _peak_f_for(X_raw: np.ndarray, ecfg: EnvelopeConfig, verbose: bool = False) -> np.ndarray:
    return peak_frequencies_batch_safe(X_raw, ecfg, verbose=verbose)


def peak_frequencies_batch_safe(X_raw: np.ndarray, ecfg: EnvelopeConfig,
                                verbose: bool = False) -> np.ndarray:
    out = np.zeros((X_raw.shape[0], ecfg.n_peaks), dtype=np.float32)
    for i in range(X_raw.shape[0]):
        out[i] = peak_frequencies(X_raw[i], ecfg, ecfg.n_peaks)
        if verbose and (i + 1) % 500 == 0:
            print('    peak_freq %d/%d' % (i + 1, X_raw.shape[0]))
    return out


def build_pools(cfg: dict, verbose: bool = True) -> Dict[str, WindowPool]:
    """构建 train / val / test（0hp 同工况）与 cross_1/2/3（1/2/3hp 跨工况）窗口池。

    - 0hp 每条记录按时间块 60/20/20 切成 train/val/test（互斥、无重叠）。
    - 跨工况池使用目标负载的**整条记录**（训练全程不可见）。
    - 逐文件按 ``max_windows_per_file`` 等间隔截断，控制类别失衡。
    """
    raw_dir = cfg['data']['raw_dir']
    window = int(cfg['data']['window'])
    stride_train = int(cfg['data']['stride_train'])
    stride_eval = int(cfg['data']['stride_eval'])
    max_per_file = int(cfg['data'].get('max_windows_per_file', 0))
    ecfg = EnvelopeConfig.from_cfg(cfg)
    classes = list(cfg['data']['classes'])
    lmap = label_map(classes)          # 标签由**本次配置的类别列表**决定，勿用默认 LABEL_MAP
    train_loads = list(cfg['data']['train_loads'])

    pools: Dict[str, Dict[str, list]] = {
        'train': {'X': [], 'y': [], 'pk': [], 'load': []},
        'val':   {'X': [], 'y': [], 'pk': [], 'load': []},
        'test':  {'X': [], 'y': [], 'pk': [], 'load': []},
    }
    for hp in cfg['data']['cross_loads']:
        pools['cross_%d' % hp] = {'X': [], 'y': [], 'pk': [], 'load': []}

    counts_log = []
    for cls in classes:
        label = lmap[cls]
        # ---- 源域（0hp）：按时间块切 train/val/test ----
        for hp in train_loads:
            code = FILE_CODES[cls][hp]
            sig, rec = load_de_signal(code, raw_dir, verbose=False)
            blocks = split_blocks(sig.size, (0.6, 0.2, 0.2))
            for name, stride, (s, e) in zip(['train', 'val', 'test'],
                                            [stride_train, stride_eval, stride_eval], blocks):
                seg = sig[s:e]
                Xr = segment(seg, window, stride)
                Xr, kept = cap_windows(Xr, max_per_file)
                if Xr.shape[0] == 0:
                    continue
                Xz = zscore_windows(Xr)
                pk = _peak_f_for(Xr, ecfg)
                pools[name]['X'].append(Xz)
                pools[name]['pk'].append(pk)
                pools[name]['y'].append(np.full(Xz.shape[0], label, dtype=np.int64))
                pools[name]['load'].append(np.full(Xz.shape[0], hp, dtype=np.int64))
                counts_log.append(dict(pool=name, cls=cls, load=hp, code=code,
                                       n_raw=int(np.asarray(sig[s:e]).size),
                                       n_win=int(Xz.shape[0]), n_kept_cap=int(len(kept))))
        # ---- 目标域（1/2/3hp）：整条记录，训练不可见 ----
        for hp in cfg['data']['cross_loads']:
            code = FILE_CODES[cls][hp]
            sig, rec = load_de_signal(code, raw_dir, verbose=False)
            Xr = segment(sig, window, stride_eval)
            Xr, kept = cap_windows(Xr, max_per_file)
            if Xr.shape[0] == 0:
                continue
            Xz = zscore_windows(Xr)
            pk = _peak_f_for(Xr, ecfg)
            key = 'cross_%d' % hp
            pools[key]['X'].append(Xz)
            pools[key]['pk'].append(pk)
            pools[key]['y'].append(np.full(Xz.shape[0], label, dtype=np.int64))
            pools[key]['load'].append(np.full(Xz.shape[0], hp, dtype=np.int64))
            counts_log.append(dict(pool=key, cls=cls, load=hp, code=code,
                                   n_raw=int(sig.size), n_win=int(Xz.shape[0]),
                                   n_kept_cap=int(len(kept))))

    out: Dict[str, WindowPool] = {}
    balance = bool(cfg['data'].get('balance_pools', False))
    for name, d in pools.items():
        if not d['X']:
            continue
        X = np.concatenate(d['X'], axis=0)
        y = np.concatenate(d['y'], axis=0)
        ld = np.concatenate(d['load'], axis=0)
        pk = np.concatenate(d['pk'], axis=0)
        if balance:
            X, y, ld, pk = _balance_classes(X, y, ld, pk, classes)
        npc = {classes[i]: int((y == i).sum()) for i in range(len(classes))}
        out[name] = WindowPool(X=X, y=y, load=ld, peak_f=pk, n_per_class=npc)

    if verbose:
        print('=== 窗口池统计（逐文件）===')
        print('%-9s %-6s %-4s %-6s %10s %8s %8s' %
              ('pool', 'class', 'hp', 'code', 'n_raw', 'n_win', 'capped_to'))
        for r in counts_log:
            print('%-9s %-6s %-4d %-6d %10d %8d %8d'
                  % (r['pool'], r['cls'], r['load'], r['code'], r['n_raw'],
                     r['n_win'], r['n_kept_cap']))
        print('=== 各池汇总%s ===' % ('（已做池内类别均衡）' if balance else ''))
        for k, v in out.items():
            sizes = list(v.n_per_class.values())
            reuse = (int(cfg['fewshot']['n_query_per_class']) *
                     int(cfg['fewshot']['n_way']) * int(cfg['eval']['test_episodes'])
                     / max(1, min(sizes)))
            print('%-9s n=%-6d ratio(max/min)=%.2f  %s  episode复用倍数≈%.0fx'
                  % (k, len(v), max(sizes) / max(1, min(sizes)), v.n_per_class, reuse))
    return out


def _balance_classes(X: np.ndarray, y: np.ndarray, ld: np.ndarray, pk: np.ndarray,
                     classes: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """把每类窗口数等间隔下采样到各类中的最小值。

    理由：CWRU 正常基线记录长度是故障记录的 2–4 倍，窗口数天然失衡。
    虽然 episode 本身是类别均衡采样的（不影响 accuracy 口径），
    但池内失衡意味着"正常类的独立形态被看见的次数更多"，
    会干扰"模型对哪一类更熟"的判断。均衡后这个混淆项被消除。
    """
    counts = [int((y == c).sum()) for c in range(len(classes))]
    target = min(c for c in counts if c > 0) if any(c > 0 for c in counts) else 0
    keep = []
    for c in range(len(classes)):
        idx = np.flatnonzero(y == c)
        if idx.size == 0:
            continue
        if idx.size > target:
            sel = np.linspace(0, idx.size - 1, target).round().astype(int)
            idx = idx[np.unique(sel)]
        keep.append(idx)
    keep = np.sort(np.concatenate(keep))
    return X[keep], y[keep], ld[keep], pk[keep]


# ----------------------------------------------------------------------
# 缓存
# ----------------------------------------------------------------------
def cache_path(cfg: dict) -> str:
    """缓存文件名带上影响切窗的所有参数指纹。

    ⚠️ 必须带指纹：否则改了 ``window / stride / cap / balance / classes`` 后
    仍会命中旧缓存，拿到的窗口池与当前配置不符 —— 这种错不报错，只让结果对不上。

    ``classes`` 尤其危险：标签是类别列表的下标，换集合 = 换标签空间，
    误复用旧缓存会让"3 类实验"悄悄跑在"4 类池"上。
    """
    d = cfg['data']
    cls = list(d.get('classes', CLASS_ORDER))
    # 可读写法（便于人工核对缓存归属）；'+' 作类间分隔，避免与字段分隔符 '_'/'-' 混淆
    cls_tag = 'c%d-%s' % (len(cls), '+'.join(cls))
    key = 'w%d_st%d_se%d_cap%d_bal%d_%s' % (
        int(d['window']), int(d['stride_train']), int(d['stride_eval']),
        int(d.get('max_windows_per_file', 0)), int(bool(d.get('balance_pools', False))),
        cls_tag)
    return os.path.join(d['processed_dir'], 'pools_%s.npz' % key)


def save_pools(pools: Dict[str, WindowPool], cfg: dict) -> str:
    os.makedirs(cfg['data']['processed_dir'], exist_ok=True)
    p = cache_path(cfg)
    arrs = {}
    for name, wp in pools.items():
        arrs['%s__X' % name] = wp.X
        arrs['%s__y' % name] = wp.y
        arrs['%s__load' % name] = wp.load
        arrs['%s__pk' % name] = wp.peak_f
        arrs['%s__npc' % name] = np.array(json.dumps(wp.n_per_class))
    np.savez_compressed(p, **arrs)
    return p


def load_pools(cfg: dict) -> Dict[str, WindowPool]:
    p = cache_path(cfg)
    if not os.path.exists(p):
        raise FileNotFoundError('未找到缓存 %s，请先调用 prepare()' % p)
    z = np.load(p, allow_pickle=False)
    names = sorted({k.split('__')[0] for k in z.files})
    out: Dict[str, WindowPool] = {}
    for n in names:
        out[n] = WindowPool(X=z['%s__X' % n], y=z['%s__y' % n],
                            load=z['%s__load' % n], peak_f=z['%s__pk' % n],
                            n_per_class=json.loads(str(z['%s__npc' % n])))
    return out


def prepare(cfg: dict, force: bool = False, verbose: bool = True) -> Dict[str, WindowPool]:
    """构建（或读取缓存）全部窗口池，并落盘 processed/pools.npz。"""
    if not force and os.path.exists(cache_path(cfg)):
        if verbose:
            print('使用缓存 %s' % cache_path(cfg))
        return load_pools(cfg)
    if verbose:
        print('--- 读取并构建窗口池（首次较慢，会写入缓存）---')
    pools = build_pools(cfg, verbose=verbose)
    p = save_pools(pools, cfg)
    if verbose:
        print('已写入缓存: %s (%.2f MB)' % (p, os.path.getsize(p) / 1e6))
    return pools


if __name__ == '__main__':
    import sys, yaml
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cfg_path = os.path.join(root, 'code', 'configs', 'default.yaml')
    with open(cfg_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg['data']['raw_dir'] = os.path.join(root, 'data', 'raw')
    cfg['data']['processed_dir'] = os.path.join(root, 'data', 'processed')

    args = sys.argv[1:]
    if '--prepare' in args:
        pools = prepare(cfg, force=('--force' in args), verbose=True)
        print('\n=== 各池汇总 ===')
        for k, v in pools.items():
            print('%-9s %s' % (k, v.summary()))
        print('\n=== 类别分布（检查是否有 4:1 失衡）===')
        for k, v in pools.items():
            tot = sum(v.n_per_class.values())
            frac = {c: '%.1f%%' % (100.0 * n / tot) for c, n in v.n_per_class.items()}
            ratio = max(v.n_per_class.values()) / max(1, min(v.n_per_class.values()))
            print('%-9s n=%-5d max/min=%.2f  %s' % (k, len(v), ratio, frac))
    else:
        print('=== 官方编号对照与文件读取 ===')
        recs = build_raw_manifest(cfg, verbose=True)
        print('\n共 %d 条记录，读取成功 %d 条' % (len(recs), sum(1 for r in recs if r.path)))
        print('\n提示：加 --prepare 构建窗口池并写入 data/processed/pools.npz')
