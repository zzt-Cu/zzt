# -*- coding: utf-8 -*-
"""轴承故障特征频率（BPFO / BPFI / BSF / FTF）及其谐波。

物理模型
--------
滚动轴承的局部缺陷（内圈剥落、外圈剥落、滚动体点蚀）在轴每转一圈时会产生
确定次数的冲击，冲击频率由几何参数与转速唯一决定：

    BPFO = (N_B / 2) * f_r * (1 - (D_B / D_P) * cos(beta))          # 外圈通过频率
    BPFI = (N_B / 2) * f_r * (1 + (D_B / D_P) * cos(beta))          # 内圈通过频率
    BSF  = (D_P / (2*D_B)) * f_r * (1 - ((D_B/D_P) * cos(beta))**2) # 滚动体自转频率
    FTF  = 0.5 * f_r * (1 - (D_B / D_P) * cos(beta))                # 保持架频率

其中 f_r 为轴转频 (Hz)，N_B 为滚珠数，D_B 为滚珠直径，D_P 为节径。

⚠️ 关于 BSF 的 2 倍约定（易错点，已交叉校验）
----------------------------------------------
上式为文献通用定义。CWRU 官方 Bearing Data Center 页面给出的故障频率倍数是
    BPFO 3.5848, BPFI 5.4152, BSF 4.7135, FTF 0.3983  (× f_r)
对本数据集轴承（6205-2RS JEM SKF）而言：
    BPFO/N_B... 代入 N_B=9, D_B=7.94mm, D_P=39.04mm, beta=0 得 3.5848 与 5.4152，与官方一致；
    而 BSF 公式给出 2.3568 —— 恰好是官方值 4.7135 的一半。
原因：滚动体缺陷在滚珠自转一周内会同时刮擦内、外圈各一次，冲击率是自转频率的 2 倍。
本模块把两者都显式给出：
    ``BSF``       —— 通用公式值（≈2.3568 f_r）
    ``BSF_2x``    —— 冲击率 / CWRU 官方口径（≈4.7135 f_r）
预训练频率软约束使用哪一个必须在论文里写清楚；本仓库默认使用 ``BSF_2x``，
因为包络谱中观测到的谱峰对应的是**冲击率**而非自转频率。见 tests/test_loader.py
中的自检：用官方倍数反算几何一致性。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ----------------------------------------------------------------------
# CWRU 12kHz 驱动端轴承几何参数（6205-2RS JEM SKF，深沟球轴承）
# 来源：CWRU Bearing Data Center 官方参数表
# ----------------------------------------------------------------------
N_B = 9              # 滚珠数
D_B_MM = 7.94        # 滚珠直径 (mm)  = 0.3126 in
D_P_MM = 39.04       # 节径     (mm)  = 1.537  in
BETA_DEG = 0.0       # 接触角 (deg)

# CWRU 官方给出的故障频率倍数（× f_r），用于交叉校验
CWRU_OFFICIAL_MULTIPLIERS = {
    'BPFO': 3.5848,
    'BPFI': 5.4152,
    'BSF': 4.7135,       # 冲击率口径（= 2 × 通用公式值）
    'FTF': 0.3983,
}

# 各类别对应的故障特征频率名称。NC（正常）无特征频率。
CLASS_FAULT_TYPE: Dict[str, Optional[str]] = {
    'NC': None,
    'IR007': 'BPFI',     # 内圈故障 -> 内圈通过频率
    'OR007': 'BPFO',     # 外圈故障 -> 外圈通过频率
    'B007': 'BSF',       # 滚动体故障 -> 滚动体冲击率
}


def shaft_freq(rpm: float) -> float:
    """轴转频 (Hz)。"""
    return float(rpm) / 60.0


def _ratio() -> float:
    return D_B_MM / D_P_MM


def bpfo(fr: float, n_b: int = N_B, d_b: float = D_B_MM, d_p: float = D_P_MM,
         beta_deg: float = BETA_DEG) -> float:
    """外圈通过频率 BPFO (Hz)。"""
    return (n_b / 2.0) * fr * (1.0 - (d_b / d_p) * math.cos(math.radians(beta_deg)))


def bpfi(fr: float, n_b: int = N_B, d_b: float = D_B_MM, d_p: float = D_P_MM,
         beta_deg: float = BETA_DEG) -> float:
    """内圈通过频率 BPFI (Hz)。"""
    return (n_b / 2.0) * fr * (1.0 + (d_b / d_p) * math.cos(math.radians(beta_deg)))


def bsf(fr: float, n_b: int = N_B, d_b: float = D_B_MM, d_p: float = D_P_MM,
        beta_deg: float = BETA_DEG) -> float:
    """滚动体自转频率 BSF (Hz)，通用公式口径。"""
    r = (d_b / d_p) * math.cos(math.radians(beta_deg))
    return (d_p / (2.0 * d_b)) * fr * (1.0 - r * r)


def bsf_2x(fr: float, **kw) -> float:
    """滚动体缺陷冲击率 = 2 × BSF，与 CWRU 官方倍数表同口径。"""
    return 2.0 * bsf(fr, **kw)


def ftf(fr: float, n_b: int = N_B, d_b: float = D_B_MM, d_p: float = D_P_MM,
        beta_deg: float = BETA_DEG) -> float:
    """保持架频率 FTF (Hz)。"""
    return 0.5 * fr * (1.0 - (d_b / d_p) * math.cos(math.radians(beta_deg)))


def fault_frequencies(rpm: float) -> Dict[str, float]:
    """给定转速，返回全部特征频率（Hz）。"""
    fr = shaft_freq(rpm)
    return {
        'f_r': fr,
        'BPFO': bpfo(fr),
        'BPFI': bpfi(fr),
        'BSF': bsf(fr),
        'BSF_2x': bsf_2x(fr),
        'FTF': ftf(fr),
    }


def harmonics(f0: float, n: int = 3, include_zero: bool = False) -> List[float]:
    """f0 的前 n 阶谐波（不含 0 阶，除非 include_zero=True）。"""
    if n < 1:
        raise ValueError('n must be >= 1')
    out = [0.0] if include_zero else []
    out.extend(h * f0 for h in range(1, n + 1))
    return out


@dataclass
class ClassFrequencySpec:
    """某个故障类别的理论频率规格。"""
    cls: str
    fault_type: Optional[str]
    fundamental: Optional[float]     # Hz，NC 为 None
    harmonics: List[float]           # Hz，长度 = n_harmonics

    def as_dict(self) -> dict:
        return asdict(self)


def class_frequency_specs(rpm: float, classes: Sequence[str] = ('NC', 'IR007', 'OR007', 'B007'),
                          n_harmonics: int = 3, use_bsf_2x: bool = True) -> Dict[str, ClassFrequencySpec]:
    """构造 类别 -> 理论频率规格 的映射。

    NC 类别 fundamental=None、harmonics=[]，对应"偏差项置零"。
    """
    fr = shaft_freq(rpm)
    table = {
        'BPFO': bpfo(fr),
        'BPFI': bpfi(fr),
        'BSF': bsf_2x(fr) if use_bsf_2x else bsf(fr),
    }
    out: Dict[str, ClassFrequencySpec] = {}
    for c in classes:
        ft = CLASS_FAULT_TYPE.get(c, None)
        if ft is None:
            out[c] = ClassFrequencySpec(cls=c, fault_type=None, fundamental=None, harmonics=[])
        else:
            f0 = table[ft]
            out[c] = ClassFrequencySpec(cls=c, fault_type=ft, fundamental=f0,
                                        harmonics=harmonics(f0, n_harmonics))
    return out


def theoretical_frequency_matrix(rpm: float, classes: Sequence[str], n_harmonics: int = 3,
                                 use_bsf_2x: bool = True) -> Tuple[List[List[float]], List[bool]]:
    """返回 (H, C) 谐波频率矩阵与有效掩码。

    - ``matrix[h][c]`` : 类别 c 的第 (h+1) 阶谐波频率 Hz；NC 列为 0 占位。
    - ``mask[c]``      : 类别 c 是否有物理频率（NC 为 False -> 偏差项置零）。

    这样损失里可以直接做 ``|f - H|`` 的张量化计算，NC 列乘 mask 后自动归零。
    """
    specs = class_frequency_specs(rpm, classes, n_harmonics, use_bsf_2x)
    matrix: List[List[float]] = []
    for h in range(n_harmonics):
        row = []
        for c in classes:
            spec = specs[c]
            row.append(spec.harmonics[h] if spec.harmonics else 0.0)
        matrix.append(row)
    mask = [specs[c].fundamental is not None for c in classes]
    return matrix, mask


def selfcheck(verbose: bool = False) -> Dict[str, float]:
    """用 CWRU 官方倍数表反算本模块实现是否一致。

    返回各类别相对误差（%），全部 < 0.5% 视为通过。
    """
    fr = 1.0     # 归一化：以 1 Hz 轴转频为基准，直接得到倍数
    got = {
        'BPFO': bpfo(fr),
        'BPFI': bpfi(fr),
        'BSF': bsf_2x(fr),      # 官方口径 = 冲击率
        'FTF': ftf(fr),
    }
    err = {}
    for k, official in CWRU_OFFICIAL_MULTIPLIERS.items():
        rel = abs(got[k] - official) / official * 100.0
        err[k] = round(rel, 4)
        if verbose:
            print('%-5s ours=%8.5f  official=%8.5f  rel_err=%.4f%%' % (k, got[k], official, rel))
    return err


if __name__ == '__main__':
    print('=== CWRU 12kHz 驱动端轴承几何 ===')
    print('N_B=%d  D_B=%.2f mm  D_P=%.2f mm  beta=%.1f deg  D_B/D_P=%.6f'
          % (N_B, D_B_MM, D_P_MM, BETA_DEG, _ratio()))
    print('\n=== 与 CWRU 官方故障频率倍数表交叉校验 ===')
    e = selfcheck(verbose=True)
    worst = max(e.values())
    print('max relative error = %.4f%%  -> %s' % (worst, 'PASS' if worst < 0.5 else 'FAIL'))

    print('\n=== 各载荷下的特征频率 (Hz) ===')
    for hp, rpm in {0: 1797, 1: 1772, 2: 1750, 3: 1730}.items():
        ff = fault_frequencies(rpm)
        print('%dhp  rpm=%d  f_r=%.2f  BPFO=%.2f  BPFI=%.2f  BSF=%.2f  BSF_2x=%.2f  FTF=%.2f'
              % (hp, rpm, ff['f_r'], ff['BPFO'], ff['BPFI'], ff['BSF'], ff['BSF_2x'], ff['FTF']))

    print('\n=== 类别 -> 理论频率 (0hp) ===')
    for cls, spec in class_frequency_specs(1797.0).items():
        print('%-6s fault=%-5s f0=%-8s harmonics=%s'
              % (cls, spec.fault_type, ('%.2f' % spec.fundamental) if spec.fundamental else 'None',
                 ['%.1f' % h for h in spec.harmonics]))
