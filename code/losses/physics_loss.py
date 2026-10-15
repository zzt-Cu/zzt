# -*- coding: utf-8 -*-
"""频率软约束损失 L_freq —— 本项目最核心的创新点（对应论文 5.5 节）。

定义（与项目文档逐字一致）
--------------------------
::

    L_freq = (1/N) * sum_i sum_c  p_{i,c} * min_k min_h  |f_{i,k} - h * f_theory(c)| / (f_theory(c) + eps)

其中
- ``N``        : batch 内样本数（query 样本）
- ``i``        : 样本下标
- ``c``        : 类别下标（4-way → c ∈ {NC, IR007, OR007, B007}）
- ``p_{i,c}``  : ``softmax(logits_i)`` 的第 c 个分量 —— **可导**，梯度由此回传
- ``f_{i,k}``  : 样本 i 包络谱的第 k 个谱峰频率 (Hz)，训练前离线预计算，**不可导**
- ``f_theory(c)``: 类别 c 的故障特征频率 (Hz)：IR007→BPFI，OR007→BPFO，B007→BSF(冲击率)
- ``h ∈ {1,2,3}``: 谐波阶数；``k ∈ {1..K}``: 谱峰序号
- ``eps``      : 数值稳定项（默认 1e-8）

三个设计选择的理由
------------------
1. **为什么用 softmax 概率加权，而不是 argmax？**
   argmax 会产生"硬"选择：模型一旦把样本判成外圈故障，就把全部约束压到 BPFO 上，
   梯度关于其余类别恒为 0，早期错误预测无从纠正，且损失对 logits 几乎处处不可导。
   softmax 加权下每个类别都贡献一份（权重 = 预测置信度），
   梯度 ``dL/dlogits_i`` 可解析求出，模型可以通过"压低错误类别的概率"来降损，
   等价于一种由物理先验提供的**软监督**，在 few-shot 只有 20 个 support 样本时尤其重要。
2. **为什么必须按 f_theory(c) 归一化？**
   四个类别的特征频率量级不同（0 hp 下 BPFO≈107 Hz、BPFI≈162 Hz、BSF≈141 Hz）。
   若不归一化，绝对偏差会让 BPFI 类的惩罚天然更大，
   等于给类别加了不等的权重，破坏了 4-way 的对称性。归一化后
   ``|Δf| / f_theory`` 是"相对偏差"，无量纲、跨类别可比、也跨转速可比。
3. **为什么需要对 NC 类别置零？**
   正常轴承没有故障特征频率，不存在可供匹配的物理量。
   若强行给 NC 指定一个频率会注入错误的物理先验，
   反而逼模型把正常样本推向某个故障类。因此 NC 的偏差项乘 0（``valid_mask``）。

可复现性注意
------------
``f_{i,k}`` 由包络谱峰值检测得到，与网络输出无关，故在数据准备阶段一次性预计算并缓存
（``data/processed/pools.npz`` 的 ``*__pk``）。这保证同一份数据在任何 run 中看到的
谱峰完全一致，损失对比才有意义。
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 允许包内相对导入与脚本直跑两种用法
try:
    from ..data import bearing_freq as bf
except (ImportError, ValueError):                                    # pragma: no cover
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data import bearing_freq as bf

LARGE = 1.0e6      # 用于把"空缺谱峰"推到永不成为 min 的位置


# ----------------------------------------------------------------------
# 阶次域（order domain）与转速自适应
# ----------------------------------------------------------------------
# 论文升级：把"故障特征频率软约束"改写为"**故障特征阶次**软约束"。
#
#   阶次  O_c = f_theory(c) / f_r      （无量纲，与载荷无关）
#   频率  f_theory(c) = O_c · f_r
#
# ⚠️ 必须显式记录的一个代数事实（否则容易把"改名"当成"改进"）：
#
#     |f - h·O_c·f_r| / (O_c·f_r)  ≡  |f/f_r - h·O_c| / O_c
#
# 即"以 Hz 写、按 f_theory 归一化"与"以阶次写、按阶次归一化"是**同一个量**，
# 当 f_r 取同一数值时逐位相等（见 code/_probe_order_domain.py 第 0 节：
# 936 个样本上最大绝对差 6.5e-10）。因此阶次域的**真正改进不在代数式**，
# 而在 **f_r 从哪里来**：
#   · 旧做法：把各载荷的**标称转速**写死进先验（等价于"现场装了转速表"）；
#   · 新做法：由包络谱峰在**物理可行窗口**内自估 f_r（见 estimate_rpm_from_peaks），
#             并把 f_r 视为带不确定性的量（见 speed_jitter_rpm）。
# 实测（_probe_order_domain.py 第 5 节）：受限窗口内自估误差 ≤ 0.64%，
# 且正确复现"随载荷单调下降"的趋势（29.90→29.50→29.10→28.65 Hz）。

# 物理可行转速窗口（4 极异步电机 @60 Hz：同步 1800 rpm，滑差 1–5%）。
# 刻意取得很宽（±300 rpm）——它只是一个"弱到几乎不构成先验"的量级约束，
# 用来避免 f_r→0 的退化解，而不是提供各载荷的真实转速。
DEFAULT_RPM_WINDOW: Tuple[float, float] = (1500.0, 2100.0)


def orders_matrix(classes: Sequence[str], n_harmonics: int = 3,
                  use_bsf_2x: bool = True) -> np.ndarray:
    """类别特征**阶次**矩阵 ``(Hn, C)``：``O[h, c] = (h+1) * O_c``；NC 列为 0。

    阶次与转速无关，故只需算一次；这使"转速自适应"退化为"阶次表固定 + 只换 f_r"。
    """
    # ⚠️ 必须用 rpm=60（⇒ f_r = 1 Hz），使矩阵元素**就等于阶次倍数**；
    #    用 rpm=1.0 会得到 O_c/60（实测把 IR007 的 5.4152 写成 0.0903），
    #    数值上不报错，只是所有频率缩了 60 倍 —— 典型的静默错误。
    matrix, mask = bf.theoretical_frequency_matrix(60.0, list(classes), n_harmonics, use_bsf_2x)
    O = np.asarray(matrix, dtype=np.float64)
    O[:, ~np.asarray(mask, dtype=bool)] = 0.0
    return O


def estimate_rpm_from_peaks(peak_f, classes: Sequence[str], n_harmonics: int = 3,
                            use_bsf_2x: bool = True,
                            rpm_window: Tuple[float, float] = DEFAULT_RPM_WINDOW,
                            n_grid: int = 121, eps: float = 1e-12
                            ) -> Tuple[float, dict]:
    """由包络谱峰**自估**轴转速（免转速先验）。

    目标函数：在物理可行窗口内搜索 rpm，使**仅故障类**的平均最小相对阶次偏差最小::

        rpm_hat = argmin_{rpm} mean_{c∈V} min_k min_h | f_k / f_r − h·O_c | / O_c

    ⚠️ **窗口是必需的**：不设窗口时谐波随 f_r 变小而变密，密集谐波可以
    「拟合」任意峰，argmin 会退化到窗口下界（实测真值排名仅 14–34/101）。
    这不是数据问题，是目标函数本身的退化解。

    Returns
    -------
    (rpm_hat, info)  ``info`` 含 ``f_r_hat``、``obj_min``、``obj_at_true_scale``。
    """
    pk = np.asarray(peak_f, dtype=np.float64)
    if pk.ndim != 2:
        raise ValueError('peak_f 需为 (N, K)')
    O = orders_matrix(classes, n_harmonics, use_bsf_2x)                  # (Hn, C)
    fmask = O[0] > 0                                                     # (C,) 故障类
    if not fmask.any():
        raise ValueError('没有可用的故障类阶次')
    O_f = O[:, fmask]                                                    # (Hn, Cf)

    lo, hi = float(rpm_window[0]), float(rpm_window[1])
    rpm_grid = np.linspace(lo, hi, int(n_grid))
    fr_grid = rpm_grid / 60.0                                            # (G,)

    p = np.where(pk > 0, pk, LARGE).astype(np.float64)                   # (N, K)
    # (N, 1, 1, 1, K) - (1, Hn, Cf, G, 1) -> (N, Hn, Cf, G, K)
    diff = np.abs(p[:, None, None, None, :] - O_f[None, :, :, None, None] * fr_grid[None, None, None, :, None])
    # 分母 (1, Hn, Cf, G)：d_h = O_c · f_r 与峰序 k 无关，故先对 k 取 min 再除以 d_h 是合法的
    #   min_h min_k |p_k − d_h| / d_h  =  min_h [ (min_k |p_k − d_h|) / d_h ]
    denom = O_f[None, :, :, None] * fr_grid[None, None, None, :]          # (1, Hn, Cf, G)
    rel = diff.min(axis=4) / (denom + eps)
    rel = rel.min(axis=1)                                                # (N, Cf, G) min over 谐波
    obj = rel.mean(axis=(0, 1))                                          # (G,) 对样本与故障类取均值
    g = int(np.argmin(obj))
    rpm_hat = float(rpm_grid[g])
    return rpm_hat, {'f_r_hat': rpm_hat / 60.0, 'obj_min': float(obj[g]),
                     'rpm_grid_lo': lo, 'rpm_grid_hi': hi, 'n_grid': int(n_grid)}


# ----------------------------------------------------------------------
# 物理频率表
# ----------------------------------------------------------------------
def build_h_matrix(rpm: float, classes: Sequence[str], n_harmonics: int = 3,
                   use_bsf_2x: bool = True,
                   device=None, dtype=torch.float32
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """构造谐波频率矩阵。

    Returns
    -------
    H     : ``(n_harmonics, C)``，``H[h, c] = (h+1) * f_theory(c)`` (Hz)
    f0    : ``(C,)``，``f_theory(c)`` (Hz)；NC 位置为 1.0 的占位值
    valid : ``(C,)`` bool，NC 为 False（偏差项置零）
    """
    matrix, mask = bf.theoretical_frequency_matrix(rpm, list(classes), n_harmonics, use_bsf_2x)
    H = torch.tensor(matrix, dtype=dtype, device=device)                       # (H, C)
    f0 = torch.tensor([float(H[0, c]) for c in range(H.shape[1])],
                      dtype=dtype, device=device)
    valid = torch.tensor(mask, dtype=torch.bool, device=device)
    # 无效类别（NC）的基频占位置 1.0，避免除零；其贡献随后被 valid 掩掉
    f0 = torch.where(valid, f0, torch.ones_like(f0))
    return H, f0, valid


def randomize_h_matrix(H: torch.Tensor, f0: torch.Tensor,
                       jitter: float = 0.5, generator=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """随机频率约束对照（消融实验用）。

    把每个类别的基频替换为 ``f0 * U(1-jitter, 1+jitter)`` 后重建谐波表，
    保持"约束强度、形式、超参"完全不变，只破坏频率的物理正确性。
    若随机版仍能带来同等提升，说明增益并非来自物理先验 —— 这是必须排除的竞争解释。
    """
    C = f0.shape[0]
    scale = (1.0 - jitter) + (2.0 * jitter) * torch.rand(C, device=f0.device,
                                                         generator=generator)
    f0_rand = f0 * scale
    hs = torch.arange(1, H.shape[0] + 1, device=H.device, dtype=H.dtype).unsqueeze(1)
    return hs * f0_rand.unsqueeze(0), f0_rand


# ----------------------------------------------------------------------
# 核心损失
# ----------------------------------------------------------------------
def _reduce(per_sample: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == 'mean':
        return per_sample.mean()
    if reduction == 'sum':
        return per_sample.sum()
    if reduction == 'none':
        return per_sample
    raise ValueError('未知 reduction: %s' % reduction)


def frequency_soft_constraint(logits: torch.Tensor, peak_f: torch.Tensor,
                              H: torch.Tensor, f0: torch.Tensor,
                              valid: Optional[torch.Tensor] = None,
                              eps: float = 1e-8, reduction: str = 'mean',
                              return_parts: bool = False,
                              fault_only: bool = True):
    """频率软约束损失。

    Parameters
    ----------
    logits : ``(N, C)`` 分类 logits（query 样本）
    peak_f : ``(N, K)`` 每个样本包络谱的前 K 个谱峰频率 (Hz)；空缺用 0 填充
    H      : ``(Hn, C)`` 谐波频率 (Hz)
    f0     : ``(C,)`` 基频 (Hz)（NC 占位 1.0）
    valid  : ``(C,)`` bool，False 的类别贡献置零（NC）
    reduction : 'mean' | 'sum' | 'none'

    Returns
    -------
    loss 或 ``(loss, parts)``；``parts`` 含 ``p``、``rel``（每个 (i,c) 的最小相对偏差）、
    ``per_sample``，便于事故排查（"损失不降"到底卡在哪一类）。
    """
    if logits.dim() != 2:
        raise ValueError('logits 需为 (N, C)，得到 %s' % (tuple(logits.shape),))
    if peak_f.dim() != 2:
        raise ValueError('peak_f 需为 (N, K)，得到 %s' % (tuple(peak_f.shape),))
    if logits.shape[0] != peak_f.shape[0]:
        raise ValueError('logits 与 peak_f 的样本数不一致: %d vs %d'
                         % (logits.shape[0], peak_f.shape[0]))
    if logits.shape[1] != H.shape[1]:
        raise ValueError('logits 类别数 %d 与 H 的列数 %d 不一致'
                         % (logits.shape[1], H.shape[1]))

    N, K = peak_f.shape
    n_h, C = H.shape
    device = logits.device

    # 设备归一（防御性，必须有）：
    #   peak_f 由 episode 提供、通常已被搬到 GPU；而 H/f0/valid 是本 module 注册的 buffer，
    #   一旦调用方漏掉 ``loss.to(device)``，它们会留在 CPU —— 实测会触发
    #   ``Expected all tensors to be on the same device, but found at least two devices``。
    #   放在这里而不是只依赖调用方，是为了让任何调用路径都不会踩到这个坑。
    #   注：.to() 在对齐设备时返回自身，零额外开销。
    peak_f = peak_f.to(device)
    H = H.to(device)
    f0 = f0.to(device)
    if valid is not None:
        valid = valid.to(device)

    # 空缺谱峰（0）替换成 LARGE，使其永远不会成为 min，等价于"该峰不存在"
    pk = torch.where(peak_f > 0, peak_f, torch.full_like(peak_f, LARGE)).float()   # (N, K)

    # 相对偏差张量 (N, K, Hn, C) -> min over 峰、min over 谐波 -> (N, C)
    diff = (pk[:, :, None, None] - H[None, None, :, :]).abs()
    rel = diff / (f0.view(1, 1, 1, C) + eps)
    rel = rel.amin(dim=2).amin(dim=1)                                      # (N, C)

    # 无物理频率的类别（NC）必须**完全退出** softmax。
    # 旧写法是"先对全部 C 类做 softmax，再把 NC 的 rel 乘 0"，这留下一个致命漏洞：
    #   NC 既占有概率质量、代价又恒为 0 —— 优化器会发现"把所有样本都判成 NC"
    #   即可令 L_freq = 0，于是约束的最优解退化为"全判 NC"。
    #   （实测：只优化旧版 L_freq 时，与真值的一致率恰为 1/C = 0.25，
    #     而该物理先验本身的判别力约 0.69 —— 说明它是在坍缩，不是在分类。）
    # 正确做法：softmax 只在**有物理频率的类别**上归一化 —— NC 既不贡献损失、
    #   也收不到梯度；NC 的判定交给 L_CE。这从根上消除了那个退化解。
    if fault_only and valid is not None and not bool(valid.all()):
        idx = valid.to(torch.bool)
        if int(idx.sum()) == 0:
            raise ValueError('valid 全为 False，没有可施加物理约束的类别')
        p_v = F.softmax(logits[:, idx].float(), dim=1)                     # (N, C_v)
        rel_v = rel[:, idx]                                                # (N, C_v)
        per_sample = (p_v * rel_v).sum(dim=1)
        loss = _reduce(per_sample, reduction)
        if return_parts:
            return loss, {'p': p_v, 'rel': rel_v, 'per_sample': per_sample,
                          'valid_idx': idx, 'fault_only': True}
        return loss

    # ---- 旧口径（fault_only=False，仅作消融对照）----
    # 全部类别参与 softmax，NC 的 rel 乘 0。已确认会让 L_freq 退化成"全判 NC"。
    p = F.softmax(logits.float(), dim=1)                                   # (N, C)
    if valid is not None:
        rel = rel * valid.view(1, C).to(rel.dtype)
    per_sample = (p * rel).sum(dim=1)
    loss = _reduce(per_sample, reduction)
    if return_parts:
        return loss, {'p': p, 'rel': rel, 'per_sample': per_sample,
                      'expected_class_penalty': rel.mean(dim=0), 'fault_only': False}
    return loss


class FrequencySoftConstraintLoss(nn.Module):
    """nn.Module 封装。``set_rpm`` 用于跨工况时切换到目标域转速。"""

    def __init__(self, rpm: float, classes: Sequence[str], n_harmonics: int = 3,
                 eps: float = 1e-8, use_bsf_2x: bool = True,
                 nc_penalty_zero: bool = True, reduction: str = 'mean',
                 fault_only: bool = True,
                 freq_jitter: float = 0.0, jitter_seed: int = 0,
                 speed_mode: str = 'fixed', speed_jitter_rpm: float = 0.0,
                 rpm_window: Optional[Tuple[float, float]] = None,
                 speed_grid: int = 121):
        super().__init__()
        self.classes = list(classes)
        self.n_harmonics = int(n_harmonics)
        self.eps = float(eps)
        self.use_bsf_2x = bool(use_bsf_2x)
        # nc_penalty_zero 仅为向后兼容/消融保留：
        #   fault_only=True（默认，推荐）时 NC 直接退出 softmax，此项不再起作用；
        #   fault_only=False 时才回到旧口径（NC 的 rel 乘 0），会退化成"全判 NC"。
        self.nc_penalty_zero = bool(nc_penalty_zero)
        self.fault_only = bool(fault_only)
        self.reduction = reduction
        # ---- 随机频率对照（消融实验）----
        # >0 时把每个类别的基频乘 U(1-j, 1+j) 后重建谐波表：
        # 约束的强度、形式、超参、λ 调度全部不变，只破坏频率的**物理正确性**。
        # 若随机版拿到与真实版同等的提升，说明增益不是来自物理先验，而是"多一项损失=正则化"。
        # ⚠️ 必须在 _set_tables 内部应用，不能只在外层改一次 buffer：
        # ``set_rpm()``（跨工况评估会调）会重建 H/f0，在外层改会被**静默撤销**，
        # 让消融退化成"其实仍然用了真实频率"，而这种失效不报任何错。
        self.freq_jitter = float(freq_jitter)
        self.jitter_seed = int(jitter_seed)
        # ---- 转速自适应（论文升级项：阶次域软约束的"自适应"部分）----
        # 'fixed'    : 用给定 rpm（默认，等价于旧做法：先验写死标称转速）
        # 'estimate' : 每个 episode 由 query 谱峰在 rpm_window 内**自估** f_r，
        #              完全不使用标称转速 ⇒ 免转速表（部署友好），且免疫标称偏差。
        #              代价：目标函数依赖 f_r 的取值，退化解由窗口兜住（见 estimate_rpm_from_peaks）。
        # 'estimate_decoupled':
        #      端到端解耦干预（用于检验 3.7 节"数据耦合"机制）。
        #      与 'estimate' **完全相同的估计器、相同的 f_r 精度分布**，
        #      但每个 episode 实际建表用的 f_r 取自**上一个 episode** 的自估值（滞后一批）。
        #      相邻 episode 是从池里独立抽取的，故该 f_r 与当前 episode 的谱峰独立 ⇒ 耦合断开，
        #      而 f_r 本身的精度（分布、偏差量级）不变。
        #      判读：若精度回到臂①水平 ⇒ 折损来自耦合；若仍等于臂③ ⇒ 耦合解释被证伪。
        # 注：阶次域本身不需要额外开关 —— 见 orders_matrix 上方关于"阶次 ≡ Hz"的代数说明。
        self.speed_mode = str(speed_mode)
        self.speed_jitter_rpm = float(speed_jitter_rpm)
        self.rpm_window = tuple(rpm_window) if rpm_window is not None else DEFAULT_RPM_WINDOW
        self.speed_grid = int(speed_grid)
        self.last_speed: Dict[str, float] = {}
        self._rpm_cache: Optional[float] = None   # 仅 estimate_decoupled 使用（滞后一批）
        self._set_tables(rpm)

    # ------------------------------------------------------------------
    def _make_tables(self, rpm: float):
        """构造某转速下的 ``(H, f0, valid)``，不注册 buffer（供转速自适应模式逐 episode 复用）。"""
        H, f0, valid = build_h_matrix(rpm, self.classes, self.n_harmonics, self.use_bsf_2x)
        if not self.nc_penalty_zero:
            valid = torch.ones_like(valid)
        if getattr(self, 'freq_jitter', 0.0) > 0:
            # 固定 seed ⇒ 每次 _set_tables（含 set_rpm 触发的）都得到**同一张**随机频率表，
            # 否则跨工况评估时各转速会看到不同的先验，实验不可比。
            g = torch.Generator(device='cpu').manual_seed(self.jitter_seed)
            H, f0 = randomize_h_matrix(H.cpu(), f0.cpu(), jitter=self.freq_jitter, generator=g)
            # NC 的占位列（f0=1.0、H=(h+1)*1.0）必须整体还原：
            # 随机化会把占位值也缩放，使 H 与 f0 对 NC **不自洽**（H≠(h+1)*f0）。
            # 虽然 NC 在 fault_only 口径下被 valid 掩掉、当前不影响数值，
            # 但这属于"表内部自相矛盾"，任何下游按 f0 反推谐波的地方都会踩到。
            hs = torch.arange(1, H.shape[0] + 1, dtype=H.dtype).view(-1, 1)
            H = torch.where(valid.view(1, -1), H, hs.expand_as(H))
            f0 = torch.where(valid, f0, torch.ones_like(f0))
        return H, f0, valid

    def _set_tables(self, rpm: float):
        H, f0, valid = self._make_tables(rpm)
        self.register_buffer('H', H)
        self.register_buffer('f0', f0)
        self.register_buffer('valid', valid)
        self.rpm = float(rpm)

    def set_rpm(self, rpm: float):
        """切换到另一转速（跨工况测试用：目标域用自身标称转速）。"""
        self._set_tables(rpm)
        return self

    def set_harmonics(self, n_harmonics: int):
        self.n_harmonics = int(n_harmonics)
        self._set_tables(self.rpm)
        return self

    # ------------------------------------------------------------------
    def forward(self, logits: torch.Tensor, peak_f: torch.Tensor,
                reduction: Optional[str] = None, return_parts: bool = False,
                fault_only: Optional[bool] = None):
        red = reduction or self.reduction
        fo = self.fault_only if fault_only is None else bool(fault_only)

        if self.speed_mode in ('estimate', 'estimate_decoupled'):
            # 免转速先验：由本批 query 谱峰自估 f_r（无梯度），再按该 f_r 建阶次表。
            # ⚠️ 自估只用峰位、不含标签与模型输出 ⇒ 不构成信息泄漏；
            #    且 f_r 是 detach 的常数，故 5.6 节的闭式梯度形式不变。
            rpm_hat, info = estimate_rpm_from_peaks(
                peak_f.detach().cpu().numpy(), self.classes, self.n_harmonics,
                self.use_bsf_2x, self.rpm_window, self.speed_grid, self.eps)
            if self.speed_mode == 'estimate_decoupled':
                # 解耦干预：本批的估计值留作**下一批**的建表转速（滞后一批）。
                # 首个 batch 尚无滞后值，退化为自估（仅影响 1 个 step，可忽略）。
                rpm_use = self._rpm_cache if self._rpm_cache is not None else rpm_hat
                self._rpm_cache = rpm_hat
            else:
                rpm_use = rpm_hat
            self.last_speed = {'rpm_est': rpm_use, 'rpm_self': rpm_hat,
                               **{k: float(v) for k, v in info.items()}}
            H, f0, valid = self._make_tables(rpm_use)
            return frequency_soft_constraint(logits, peak_f, H, f0, valid, self.eps,
                                             red, return_parts, fo)

        if self.speed_mode == 'jitter' and self.speed_jitter_rpm > 0:
            # 训练期对 f_r 做区间随机化：等价于"只学阶次模式、不学绝对频率网格"，
            # 使约束对加载引起的转速漂移（实测真类失配膨胀最高 +55%）不敏感。
            d = float(torch.empty(1).uniform_(-1.0, 1.0).item()) * self.speed_jitter_rpm
            rpm_j = self.rpm + d
            self.last_speed = {'rpm_jitter': rpm_j, 'rpm_base': self.rpm}
            H, f0, valid = self._make_tables(rpm_j)
            return frequency_soft_constraint(logits, peak_f, H, f0, valid, self.eps,
                                             red, return_parts, fo)

        self.last_speed = {'rpm_fixed': self.rpm}
        return frequency_soft_constraint(
            logits, peak_f, self.H, self.f0, self.valid, self.eps,
            red, return_parts=return_parts, fault_only=fo)

    @classmethod
    def from_cfg(cls, cfg: dict, rpm: Optional[float] = None) -> 'FrequencySoftConstraintLoss':
        loss_cfg = cfg['loss']
        classes = list(cfg['data']['classes'])
        hp = int(cfg['data']['train_loads'][0])
        if rpm is None:
            rpm = float(cfg['freq']['shaft_rpm'][hp])
        win = loss_cfg.get('rpm_window', None)
        return cls(rpm=rpm, classes=classes,
                   n_harmonics=int(loss_cfg['n_harmonics']),
                   eps=float(loss_cfg['eps']),
                   nc_penalty_zero=bool(loss_cfg.get('nc_penalty_zero', True)),
                   freq_jitter=float(loss_cfg.get('freq_jitter', 0.0)),
                   jitter_seed=int(loss_cfg.get('jitter_seed', 0)),
                   speed_mode=str(loss_cfg.get('speed_mode', 'fixed')),
                   speed_jitter_rpm=float(loss_cfg.get('speed_jitter_rpm', 0.0)),
                   rpm_window=tuple(win) if win else None,
                   speed_grid=int(loss_cfg.get('speed_grid', 121)))


# ----------------------------------------------------------------------
# 总损失
# ----------------------------------------------------------------------
class TotalLoss(nn.Module):
    """``L_total = ce_weight * L_CE + lambda(t) * L_freq``"""

    def __init__(self, loss_cfg: dict, freq_loss: FrequencySoftConstraintLoss):
        super().__init__()
        self.ce_weight = float(loss_cfg.get('ce_weight', 1.0))
        self.freq = freq_loss
        self.last: Dict[str, float] = {}

    def forward(self, logits: torch.Tensor, target: torch.Tensor, peak_f: torch.Tensor,
                lam: float, return_parts: bool = False):
        l_ce = F.cross_entropy(logits, target)
        l_freq, parts = self.freq(logits, peak_f, return_parts=True)
        total = self.ce_weight * l_ce + float(lam) * l_freq
        self.last = {'loss_total': float(total.detach()), 'loss_ce': float(l_ce.detach()),
                     'loss_freq': float(l_freq.detach()), 'lambda': float(lam)}
        if return_parts:
            return total, self.last, parts
        return total


# ----------------------------------------------------------------------
if __name__ == '__main__':
    torch.manual_seed(0)
    classes = ['NC', 'IR007', 'OR007', 'B007']
    rpm = 1797.0

    print('=== 0hp 下的谐波频率表 H (Hz) ===')
    H, f0, valid = build_h_matrix(rpm, classes, 3)
    print('       ' + '  '.join('%-10s' % c for c in classes))
    for h in range(H.shape[0]):
        print('h=%d   ' % (h + 1) + '  '.join('%10.2f' % v for v in H[h].tolist()))
    print('f0    =', [round(x, 2) for x in f0.tolist()])
    print('valid =', valid.tolist())

    # ---- 构造"完美"样本：谱峰正好落在各类别基频上 ----
    N, K = 8, 5
    peak = torch.zeros(N, K)
    for i, c in enumerate([0, 1, 2, 3] * (N // 4)):
        if valid[c]:
            peak[i, 0] = f0[c]
            peak[i, 1] = 2 * f0[c]
        else:
            peak[i, 0] = 137.0        # 正常样本的谱峰是随机结构，本来就没有特征频率

    print('\n=== 情形 A：logits 接近 one-hot（模型很确信）时 L_freq 应趋 0 ===')
    lg_conf = torch.zeros(N, 4)
    for i, c in enumerate([0, 1, 2, 3] * (N // 4)):
        lg_conf[i, c] = 30.0            # 置信度极高
    l1, parts1 = frequency_soft_constraint(lg_conf, peak, H, f0, valid, return_parts=True)
    print('L_freq = %.3e' % float(l1.detach()))
    print('各样本 =', ['%.3e' % float(x) for x in parts1['per_sample']])
    print('  -> %s' % ('PASS' if float(l1) < 1e-6 else 'FAIL'))

    print('\n=== 情形 B：logits 随机（模型还没学会）时 L_freq 明显大于 0 ===')
    lg_rand = torch.randn(N, 4)
    l2 = frequency_soft_constraint(lg_rand, peak, H, f0, valid)
    print('L_freq = %.4f  -> %s' % (float(l2),
          'PASS' if float(l2) > float(l1) else 'FAIL'))
    print('  说明：L_freq 是对**所有类别**按 softmax 概率加权求和，'
          '即使谱峰完美匹配真值，只要模型把概率分给了别的类别，惩罚就还在 '
          '—— 这正是"软约束"的含义，也是梯度能纠正错误预测的来源。')

    # ---- NC 置零验证：让 NC 类别的 logits 占绝对优势 ----
    print('\n=== NC 偏差项置零验证 ===')
    pk_nc = torch.full((4, K), 0.0)
    pk_nc[:, 0] = 137.0
    lg_nc = torch.zeros(4, 4)
    lg_nc[:, 0] = 20.0                    # 几乎 100% 判为 NC
    lz = frequency_soft_constraint(lg_nc, pk_nc, H, f0, valid)
    print('NC 占优时的 L_freq = %.3e' % float(lz))
    print('  残余量级来源：softmax 在 3 个故障类上仍留有 ~1e-9 的概率质量，'
          '这是数值残差而非 NC 项贡献')
    print('  -> %s' % ('PASS(< 1e-8)' if float(lz) < 1e-8 else 'FAIL'))
    lz2 = frequency_soft_constraint(lg_nc, pk_nc, H, f0,
                                    torch.ones_like(valid))
    print('若关闭 NC 置零(反例)   = %.3e  (应远大于 0)' % float(lz2))
    print('  -> %s' % ('PASS' if float(lz2) > 1.0 else 'FAIL'))

    # ---- 梯度确实回传到 logits ----
    print('\n=== 梯度回传与形状 ===')
    logits = lg_rand.clone().requires_grad_(True)
    loss = frequency_soft_constraint(logits, peak + 12.0, H, f0, valid)
    loss.backward()
    print('L_freq = %.4f' % float(loss.detach()))
    print('grad 形状 =%s  grad_norm=%.4f  finite=%s'
          % (tuple(logits.grad.shape), float(logits.grad.norm()),
             bool(torch.isfinite(logits.grad).all())))

    # ---- 随机频率对照 ----
    print('\n=== 随机频率约束对照 ===')
    Hr, f0r = randomize_h_matrix(H, f0, jitter=0.5, generator=torch.Generator().manual_seed(0))
    print('原 f0 =', [round(x, 2) for x in f0.tolist()])
    print('随机 f0=', [round(x, 2) for x in f0r.tolist()])
    lr = frequency_soft_constraint(logits.detach().clone(), peak, Hr, f0r, valid)
    print('随机频率下的 L_freq = %.4f' % float(lr.detach()))

    # ---- 谐波阶数消融（仅基频） ----
    H1, f01, v1 = build_h_matrix(rpm, classes, 1)
    l_only1 = frequency_soft_constraint(logits.detach().clone(), peak, H1, f01, v1)
    H3, f03, v3 = build_h_matrix(rpm, classes, 3)
    l_1to3 = frequency_soft_constraint(logits.detach().clone(), peak, H3, f03, v3)
    print('\n=== 仅基频 vs 基频+谐波 的损失口径对比 ===')
    print('n_harmonics=1: %.4f   n_harmonics=3: %.4f' % (float(l_only1), float(l_1to3)))
