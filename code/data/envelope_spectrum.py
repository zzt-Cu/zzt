# -*- coding: utf-8 -*-
"""包络谱提取与谱峰检测。

链路（与项目文档一致）
----------------------
原始振动信号 -> Butterworth 带通 (500-5000 Hz) -> Hilbert 解析信号取模得包络
            -> 去趋势/去均值                -> rFFT 得包络谱 -> 谱峰检测取前 K 个

为什么需要包络分析
------------------
轴承局部缺陷产生的冲击是**宽带瞬态**，其能量被结构共振（数千 Hz）调制，
直接在原始频谱上看不到 BPFO/BPFI/BSF 的谱线；只有把共振带解调出来
（Hilbert 包络）后，故障特征频率及其谐波才会以清晰的谱峰形式出现在**低频段**。
这就是频率软约束能够作为"物理先验"注入的物理基础。

实现要点
--------
1. 滤波用 ``sosfiltfilt``（零相位、双向），避免相位失真导致的包络畸变。
2. 谱峰定位用零填充 FFT（``n_fft`` > 窗长）。零填充不增加信息量，只是把
   离散谱线插值到更细的频率栅格上，用于**提高峰值频率的读数精度**；
   这一点在论文中必须如实说明，不能宣称提高了频率分辨率。
3. 峰值检测用 ``scipy.signal.find_peaks``，带 ``distance``（最小频率间隔）
   与相对幅度阈值，避免同一谱峰被重复计入。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt


# ----------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------
@dataclass
class EnvelopeConfig:
    fs: int = 12000
    band: Tuple[float, float] = (500.0, 5000.0)
    order: int = 4
    n_fft: int = 8192              # 零填充长度，仅用于插值谱线
    detrend: bool = True           # 包络去线性趋势
    remove_mean: bool = True       # 去掉包络直流分量
    f_min: float = 10.0            # 忽略低于该频率的谱线（DC 泄漏区）
    n_peaks: int = 5               # K：保留的谱峰数
    peak_min_sep_hz: float = 10.0  # 谱峰最小间隔
    peak_min_rel_height: float = 0.05   # 相对最大幅值的下界
    window: str = 'hann'

    @staticmethod
    def from_cfg(cfg: dict, fs: Optional[int] = None, n_peaks: Optional[int] = None) -> 'EnvelopeConfig':
        """从项目 YAML 配置构造。``cfg`` 为已加载的 dict。"""
        f = cfg['freq']
        return EnvelopeConfig(
            fs=fs if fs is not None else int(cfg['data']['fs']),
            band=tuple(float(x) for x in f['band']),
            order=int(f['filter_order']),
            n_peaks=int(n_peaks if n_peaks is not None else f['n_peaks']),
            peak_min_sep_hz=float(f['peak_min_sep_hz']),
        )


# ----------------------------------------------------------------------
# 基础算子
# ----------------------------------------------------------------------
def bandpass(sig: np.ndarray, fs: float, band: Tuple[float, float], order: int = 4,
             zero_phase: bool = True) -> np.ndarray:
    """Butterworth 带通滤波。band 需满足 0 < lo < hi < fs/2。"""
    nyq = fs / 2.0
    lo, hi = float(band[0]), float(band[1])
    if not (0 < lo < hi < nyq):
        raise ValueError('band=%s 非法：需满足 0 < lo < hi < fs/2=%.1f' % (str(band), nyq))
    sos = butter(order, [lo / nyq, hi / nyq], btype='band', output='sos')
    if zero_phase:
        return sosfiltfilt(sos, sig)
    return sosfiltfilt(sos, sig)


def envelope(sig: np.ndarray) -> np.ndarray:
    """Hilbert 包络（解析信号取模）。"""
    return np.abs(hilbert(np.asarray(sig, dtype=np.float64)))


def envelope_spectrum(sig: np.ndarray, cfg: EnvelopeConfig,
                      return_complex: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """完整链路：带通 -> 包络 -> 去趋势/去均值 -> 单边幅度谱。

    返回 ``(freqs, amp)``，长度均为 ``n_fft // 2 + 1``。
    """
    x = np.asarray(sig, dtype=np.float64).ravel()
    if x.size < 32:
        raise ValueError('信号过短：%d' % x.size)
    xf = bandpass(x, cfg.fs, cfg.band, cfg.order)
    env = envelope(xf)
    if cfg.detrend:
        # 去线性趋势，抑制包络中由转速波动引起的低频漂移
        n = env.size
        t = np.arange(n, dtype=np.float64)
        a, b = np.polyfit(t, env, 1)
        env = env - (a * t + b)
    if cfg.remove_mean:
        env = env - env.mean()

    n_fft = max(int(cfg.n_fft), env.size)
    w = np.hanning(env.size) if cfg.window == 'hann' else np.ones(env.size)
    spec = np.fft.rfft(env * w, n=n_fft)
    amp = np.abs(spec) * (2.0 / (w.sum() + 1e-12))    # 幅度归一化，便于跨样本比较
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / cfg.fs)
    return freqs, amp.astype(np.float64)


def detect_peaks(freqs: np.ndarray, amp: np.ndarray, cfg: EnvelopeConfig,
                 n_peaks: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """谱峰检测。返回 ``(peak_freqs, peak_amps, peak_idx)``，按幅度降序、长度 <= n_peaks。"""
    k = int(n_peaks if n_peaks is not None else cfg.n_peaks)
    df = freqs[1] - freqs[0]
    distance = max(1, int(round(cfg.peak_min_sep_hz / df)))
    lo_idx = int(np.searchsorted(freqs, cfg.f_min))
    band_amp = amp[lo_idx:]
    if band_amp.size == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0, dtype=int)
    thr = cfg.peak_min_rel_height * float(band_amp.max())
    idx, props = find_peaks(band_amp, height=thr, distance=distance)
    if idx.size == 0:
        # 退化情形：直接取最大点
        idx = np.array([int(np.argmax(band_amp))])
    idx = idx + lo_idx
    order = np.argsort(-amp[idx])[:k]
    idx = idx[order]
    return freqs[idx].copy(), amp[idx].copy(), idx.astype(int)


def peak_frequencies(sig: np.ndarray, cfg: EnvelopeConfig,
                     n_peaks: Optional[int] = None) -> np.ndarray:
    """只取前 K 个谱峰的频率值（Hz），**按幅度降序**，不足 K 个时用 0 补齐。

    该函数是频率软约束损失的输入（不可导，训练前离线预计算）。
    """
    k = int(n_peaks if n_peaks is not None else cfg.n_peaks)
    f, a, _ = detect_peaks(*envelope_spectrum(sig, cfg), cfg, k)
    out = np.zeros(k, dtype=np.float32)
    m = min(k, f.size)
    out[:m] = f[:m]
    return out


def peak_frequencies_batch(X: np.ndarray, cfg: EnvelopeConfig,
                           n_peaks: Optional[int] = None,
                           verbose_every: int = 0) -> np.ndarray:
    """批量求谱峰频率。``X``: (N, L) -> (N, K)。"""
    k = int(n_peaks if n_peaks is not None else cfg.n_peaks)
    out = np.zeros((X.shape[0], k), dtype=np.float32)
    for i in range(X.shape[0]):
        out[i] = peak_frequencies(X[i], cfg, k)
        if verbose_every and (i + 1) % verbose_every == 0:
            print('  peak_frequencies_batch %d/%d' % (i + 1, X.shape[0]))
    return out


# ----------------------------------------------------------------------
# 诊断用工具
# ----------------------------------------------------------------------
def harmonic_ratios(peak_f: np.ndarray, f_theory: float) -> np.ndarray:
    """把谱峰频率换算成"理论频率的倍数"，便于人工核对 BPFO/BPFI/BSF 及其谐波。"""
    if f_theory <= 0:
        return np.zeros_like(peak_f)
    return np.asarray(peak_f, dtype=np.float64) / f_theory


def nearest_harmonic_error(peak_f: np.ndarray, f_theory: float,
                           n_harmonics: int = 3) -> float:
    """最小相对谐波偏差 min_k min_h |f_k - h*f0| / f0。与损失里的口径完全一致。"""
    pf = np.asarray(peak_f, dtype=np.float64).ravel()
    pf = pf[pf > 0]
    if pf.size == 0 or f_theory <= 0:
        return float('nan')
    hs = np.arange(1, n_harmonics + 1, dtype=np.float64) * f_theory
    d = np.abs(pf[:, None] - hs[None, :]).min() / f_theory
    return float(d)


def selfcheck(sig: np.ndarray, f_theory: float, cfg: EnvelopeConfig,
              n_harmonics: int = 3, top: int = 3) -> Dict[str, object]:
    """对单条信号做一次可读的自检：打印前若干谱峰及其相对理论频率的阶数。"""
    f, a, _ = detect_peaks(*envelope_spectrum(sig, cfg), cfg)
    ratios = harmonic_ratios(f, f_theory)
    return {
        'peak_freqs': f.tolist(),
        'peak_amps': a.tolist(),
        'harmonic_ratios': ratios.round(4).tolist(),
        'min_rel_err': nearest_harmonic_error(f, f_theory, n_harmonics),
        'top_ratios': ratios[:top].round(4).tolist(),
    }


if __name__ == '__main__':
    # 合成信号冒烟测试：单频调幅（载波 3 kHz，调制 BPFO=107.4 Hz）
    fs, n, f_mod, f_car = 12000, 2048, 107.37, 3000.0
    t = np.arange(n) / fs
    rng = np.random.default_rng(0)
    sig = (1.0 + 0.8 * np.cos(2 * np.pi * f_mod * t)) * np.cos(2 * np.pi * f_car * t)
    sig = sig + 0.05 * rng.standard_normal(n)
    cfg = EnvelopeConfig(fs=fs, band=(500, 5000), order=4, n_fft=8192, n_peaks=5)
    info = selfcheck(sig, f_mod, cfg)
    print('=== 合成信号自检（理论 f0=%.2f Hz）===' % f_mod)
    print('peak_freqs      :', [round(x, 2) for x in info['peak_freqs']])
    print('peak_amps       :', [round(x, 4) for x in info['peak_amps']])
    print('harmonic_ratios :', info['harmonic_ratios'])
    print('min_rel_err     : %.5f' % info['min_rel_err'])
    ok = info['min_rel_err'] < 0.02
    print('RESULT: %s' % ('PASS' if ok else 'FAIL'))
