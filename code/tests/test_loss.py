# -*- coding: utf-8 -*-
"""损失层测试：λ 调度时序、频率软约束的解析性质、NC 置零、梯度、消融开关。

这里的重点不是"能跑"，而是验证 L_freq 的**数学性质**与文档定义严格一致。
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from losses import loss_scheduler as ls                      # noqa: E402
from losses import physics_loss as pl                        # noqa: E402

CLASSES = ['NC', 'IR007', 'OR007', 'B007']
RPM = 1797.0


# ======================================================================
# λ(t) 调度
# ======================================================================
def test_stepwise_matches_documented_sequence_exactly():
    """文档规定：前 20 轮 λ=0，之后每 10 轮 +0.1 至 0.5。逐轮比对。"""
    s = ls.StepwiseLambda(warmup_epochs=20, step=10, step_value=0.1, lambda_max=0.5)
    expect = ([0.0] * 20 + [0.1] * 10 + [0.2] * 10 + [0.3] * 10 + [0.4] * 10 + [0.5] * 20)
    assert s.values(80) == expect


def test_stepwise_caps_at_lambda_max():
    s = ls.StepwiseLambda(20, 10, 0.1, 0.5)
    assert s.value(200) == 0.5
    assert max(s.values(500)) == 0.5


def test_stepwise_float_precision_does_not_leak():
    """0.1*3 在浮点下是 0.30000000000000004，必须被 round 掉，
    否则日志里会出现 0.30000000000000004 这种值、与文档序列比对也会失败。"""
    s = ls.StepwiseLambda(20, 10, 0.1, 0.5)
    assert s.value(40) == 0.3
    assert repr(s.value(40)) == '0.3'


def test_fixed_and_zero_schedulers():
    assert ls.FixedLambda(0.3).values(5) == [0.3] * 5
    assert set(ls.ConstantLambda().values(50)) == {0.0}


def test_adaptive_is_monotonic_and_bounded():
    s = ls.AdaptiveLambda(20, 10, 0.5)
    v = s.values(120)
    assert all(b >= a - 1e-12 for a, b in zip(v, v[1:])), '必须单调不减'
    assert max(v) <= 0.5 + 1e-12 and min(v) >= 0.0
    assert abs(v[-1] - 0.5) < 0.02


def test_build_scheduler_from_cfg():
    cfg = {'lambda_strategy': 'stepwise', 'warmup_epochs': 20, 'step': 10,
           'step_value': 0.1, 'lambda_max': 0.5}
    assert isinstance(ls.build_lambda_scheduler(cfg), ls.StepwiseLambda)
    assert isinstance(ls.build_lambda_scheduler({'lambda_strategy': 'fixed',
                                                 'fixed_lambda': 0.2}), ls.FixedLambda)
    with pytest.raises(ValueError):
        ls.build_lambda_scheduler({'lambda_strategy': 'nope'})


# ======================================================================
# 频率软约束
# ======================================================================
@pytest.fixture
def tables():
    return pl.build_h_matrix(RPM, CLASSES, 3)


def _peaks_for_class(c: int, f0: torch.Tensor, valid: torch.Tensor, n: int,
                     k: int = 5, harmonics=(1, 2)) -> torch.Tensor:
    p = torch.zeros(n, k)
    if valid[c]:
        for j, h in enumerate(harmonics):
            p[:, j] = h * f0[c]
    else:
        p[:, 0] = 137.0                     # 正常样本谱峰是随机结构
    return p


def _class_aligned_batch(f0: torch.Tensor, valid: torch.Tensor, per_c: int,
                         confident: bool = True, offset: float = 0.0,
                         seed: int = 0, include_nc: bool = False):
    """构造一批"样本谱峰所属类 == logits 确信类"的 (peak, logits)。

    为什么需要它：L_freq = Σ_c p_c · rel_c 是按 **softmax 权重** 对各类偏差加权，
    只有"样本被判为它真正所属的类"时，这一项才会被压到 0。

    早期写法把各类峰值按 c=0,1,2,3 顺序拼起来，再用 ``logits[i, i % 4] = 30``
    指定标签 —— 于是第 1 条（类 0 的谱峰）被判成类 1，峰值与标签**错位**，
    loss 自然停在 0.12 而不是 0。这里显式按类组织，杜绝该错位。

    ``include_nc=False``（默认）只造 3 个故障类。原因：新口径下 NC 已退出
    softmax，NC 样本的 L_freq 恒为一个"背景值"（它的谱峰本就不匹配任何故障
    频率），若混进"峰值匹配则损失为 0"这类断言里会把结论搅浑。
    """
    g = torch.Generator().manual_seed(seed)
    peaks, logits = [], []
    for c in ([0, 1, 2, 3] if include_nc else [1, 2, 3]):
        peaks.append(_peaks_for_class(c, f0, valid, per_c))
        if confident:
            lg = torch.full((per_c, 4), -10.0)
            lg[:, c] = 30.0                       # argmax 恰好是 c
        else:
            lg = torch.randn(per_c, 4, generator=g)
        logits.append(lg)
    peak = torch.cat(peaks, dim=0)
    if offset:
        peak = peak + float(offset)
    return peak, torch.cat(logits, dim=0)


def test_h_matrix_matches_theory(tables):
    H, f0, valid = tables
    assert H.shape == (3, 4)
    assert valid.tolist() == [False, True, True, True]
    assert abs(float(f0[1]) - 162.19) < 0.05      # BPFI @1797rpm
    assert abs(float(f0[2]) - 107.36) < 0.05      # BPFO
    assert abs(float(f0[3]) - 141.17) < 0.05      # BSF(冲击率)
    np.testing.assert_allclose(H[1, 1].item(), 2 * H[0, 1].item(), rtol=1e-6)


def test_loss_zero_when_confident_and_peaks_match(tables):
    H, f0, valid = tables
    peak, logits = _class_aligned_batch(f0, valid, per_c=2)   # 3 个故障类 × 2 = 6 条
    assert peak.shape[0] == logits.shape[0] == 6
    loss = pl.frequency_soft_constraint(logits, peak, H, f0, valid)
    assert float(loss) < 1e-6, float(loss)


def test_loss_positive_when_peaks_mismatch(tables):
    H, f0, valid = tables
    peak, logits = _class_aligned_batch(f0, valid, per_c=2, offset=20.0)
    loss = pl.frequency_soft_constraint(logits, peak, H, f0, valid)
    assert float(loss) > 1e-3, float(loss)


def test_nc_logits_do_not_affect_loss(tables):
    """NC 完全退出 softmax：NC 列的 logits 既不影响损失、也收不到梯度。

    这是修掉"全判 NC"退化解的关键性质（见 test_legacy_nc_zero_penalty_is_degenerate）。
    """
    H, f0, valid = tables
    peak = torch.zeros(4, 5)
    peak[:, 0] = 137.0                       # NC 样本的谱峰是随机结构
    z1 = torch.zeros(4, 4)
    z1[:, 0] = 25.0
    z2 = z1.clone()
    z2[:, 0] = -25.0                         # 只改 NC 列
    l1 = pl.frequency_soft_constraint(z1, peak, H, f0, valid)
    l2 = pl.frequency_soft_constraint(z2, peak, H, f0, valid)
    assert torch.allclose(l1, l2), (float(l1), float(l2))

    z = z1.clone().requires_grad_(True)
    pl.frequency_soft_constraint(z, peak, H, f0, valid).backward()
    assert float(z.grad[:, 0].abs().sum()) == 0.0, '梯度不应流向 NC 列'
    assert float(z.grad[:, 1:].abs().sum()) > 0.0, '故障类必须仍有梯度'


def test_legacy_nc_zero_penalty_is_degenerate(tables):
    """记录旧口径的缺陷（作为修复的动机与回归防线）。

    旧实现 ``fault_only=False``：先对全部 4 类做 softmax，再把 NC 的 rel 乘 0。
    于是"把所有样本都判成 NC"可令 L_freq = 0 —— 约束的全局最优退化为坍缩。
    实测印证：只优化旧版 L_freq 时与真值的一致率恰为 1/4，而该先验本身的
    判别力约 0.69。新口径必须让这种情况不再发生。
    """
    H, f0, valid = tables
    peak = torch.zeros(4, 5)
    peak[:, 0] = 137.0
    z = torch.zeros(4, 4)
    z[:, 0] = 25.0                            # 全部判为 NC
    l_old = pl.frequency_soft_constraint(z, peak, H, f0, valid, fault_only=False)
    l_new = pl.frequency_soft_constraint(z, peak, H, f0, valid, fault_only=True)
    assert float(l_old) < 1e-8, ('旧口径下"全判 NC"损失应为 0（这就是缺陷）', float(l_old))
    assert float(l_new) > 1e-3, ('新口径下不应再存在这个免费解', float(l_new))


def test_fault_only_loss_positive_when_peaks_match(tables):
    """fault_only=True 时，故障样本峰值匹配 -> 损失趋 0；不匹配 -> 明显变大。"""
    H, f0, valid = tables
    peak, logits = _class_aligned_batch(f0, valid, per_c=2)
    l_ok = pl.frequency_soft_constraint(logits, peak, H, f0, valid, fault_only=True)
    assert float(l_ok) < 1e-6, float(l_ok)
    l_bad = pl.frequency_soft_constraint(logits, peak + 25.0, H, f0, valid,
                                         fault_only=True)
    assert float(l_bad) > 1e-3, float(l_bad)


def test_normalization_makes_loss_scale_invariant(tables):
    """按 f_theory 归一化后，把 H 与谱峰同时等比缩放，损失不变。"""
    H, f0, valid = tables
    n = 8
    peak = torch.cat([_peaks_for_class(c, f0, valid, 2) for c in range(4)], dim=0)
    logits = torch.randn(n, 4)
    l1 = pl.frequency_soft_constraint(logits, peak, H, f0, valid)
    s = 2.0
    l2 = pl.frequency_soft_constraint(logits, peak * s, H * s, f0 * s, valid)
    np.testing.assert_allclose(float(l1), float(l2), rtol=1e-5)


def test_min_over_peaks_and_harmonics(tables):
    """min_k min_h 语义：只要有**任意一个**谱峰落在**任意一阶**谐波附近，惩罚就小。"""
    H, f0, valid = tables
    c = 2                                     # OR007 -> BPFO = 107.36 Hz
    k = 5
    wrong = torch.zeros(1, k)
    wrong[0, 0] = 900.0                       # 远离所有谐波
    right = wrong.clone()
    right[0, 3] = 3 * float(f0[c])            # 第 3 阶谐波处放一个峰
    logits = torch.full((1, 4), -30.0)
    logits[0, c] = 30.0
    l_wrong = float(pl.frequency_soft_constraint(logits, wrong, H, f0, valid))
    l_right = float(pl.frequency_soft_constraint(logits, right, H, f0, valid))
    assert l_right < 1e-4, l_right
    assert l_wrong > l_right


def test_zero_peaks_are_treated_as_absent(tables):
    """空缺谱峰（0 填充）不能伪装成"匹配上了 0 Hz"。"""
    H, f0, valid = tables
    k = 5
    all_zero = torch.zeros(2, k)
    logits = torch.zeros(2, 4)
    logits[:, 1] = 30.0                       # 判为 IR007 -> BPFI
    loss = pl.frequency_soft_constraint(logits, all_zero, H, f0, valid)
    # 全部谱峰缺失 -> 相对偏差应约为 1.0（|0 - f0|/f0 被替换为 LARGE 后仍取不到 0）
    assert float(loss) > 0.5, float(loss)


def test_gradient_flows_and_is_finite(tables):
    H, f0, valid = tables
    n = 6
    peak, _ = _class_aligned_batch(f0, valid, per_c=2, offset=15.0)
    peak = peak[:n]                                          # 裁到与 logits 同长度
    # 这里**故意**用未饱和的随机 logits：若拿 [-10,…,30,…] 这种极端 one-hot，
    # softmax 的雅可比近似对角化，∂L/∂z 会缩到 ~1e-40 量级，
    # "梯度非零"的断言就退化成在检验浮点噪声，失去意义。
    g = torch.Generator().manual_seed(7)
    logits = torch.randn(n, 4, generator=g).requires_grad_(True)
    assert peak.shape[0] == logits.shape[0] == n, (peak.shape, logits.shape)
    loss = pl.frequency_soft_constraint(logits, peak, H, f0, valid)
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.shape == logits.shape
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad.norm()) > 1e-8, '梯度不应恒为 0（argmax 版就会这样）'


def test_reduction_modes(tables):
    H, f0, valid = tables
    logits = torch.randn(5, 4)
    peak = torch.zeros(5, 5)
    peak[:, 0] = 110.0
    full = pl.frequency_soft_constraint(logits, peak, H, f0, valid, reduction='none')
    assert full.shape == (5,)
    np.testing.assert_allclose(float(full.mean()),
                               float(pl.frequency_soft_constraint(logits, peak, H, f0, valid,
                                                                  reduction='mean')), rtol=1e-5)
    np.testing.assert_allclose(float(full.sum()),
                               float(pl.frequency_soft_constraint(logits, peak, H, f0, valid,
                                                                  reduction='sum')), rtol=1e-5)


def test_shape_validation(tables):
    H, f0, valid = tables
    with pytest.raises(ValueError):
        pl.frequency_soft_constraint(torch.randn(4, 5), torch.randn(4, 5), H, f0, valid)
    with pytest.raises(ValueError):
        pl.frequency_soft_constraint(torch.randn(4, 4), torch.randn(3, 5), H, f0, valid)
    with pytest.raises(ValueError):
        pl.frequency_soft_constraint(torch.randn(2, 4), torch.randn(2, 5), H, f0, valid,
                                     reduction='nope')


def test_randomize_h_matrix_stays_in_range_and_changes_values(tables):
    H, f0, valid = tables
    Hr, f0r = pl.randomize_h_matrix(H, f0, jitter=0.5,
                                    generator=torch.Generator().manual_seed(0))
    assert Hr.shape == H.shape
    assert not torch.allclose(Hr, H)
    # NC 列仍为 1.0 基准的谐波（保证形状/数值合法），且故障列在 ±50% 内
    for c in (1, 2, 3):
        assert 0.5 * float(f0[c]) - 1e-6 <= float(f0r[c]) <= 1.5 * float(f0[c]) + 1e-6


def test_harmonics_ablation_changes_reference_set(tables):
    H1, f01, v1 = pl.build_h_matrix(RPM, CLASSES, 1)
    H3, f03, v3 = pl.build_h_matrix(RPM, CLASSES, 3)
    assert H1.shape[0] == 1 and H3.shape[0] == 3
    np.testing.assert_allclose(H1[0].numpy(), H3[0].numpy(), rtol=1e-6)


def test_module_wrapper_set_rpm_shifts_frequencies():
    m = pl.FrequencySoftConstraintLoss(rpm=1797.0, classes=CLASSES, n_harmonics=3)
    f0a = m.f0.clone()
    m.set_rpm(1730.0)
    f0b = m.f0.clone()
    assert float(f0b[2]) < float(f0a[2])
    assert abs(float(f0a[2]) / float(f0b[2]) - 1797.0 / 1730.0) < 1e-5


# ----------------------------------------------------------------------
# 随机频率对照（消融）
# ----------------------------------------------------------------------
def test_freq_jitter_disabled_by_default_is_physical():
    m = pl.FrequencySoftConstraintLoss(rpm=RPM, classes=CLASSES, n_harmonics=3)
    Href, f0ref, _ = pl.build_h_matrix(RPM, CLASSES, 3)
    np.testing.assert_allclose(m.H.numpy(), Href.numpy(), rtol=1e-6)
    np.testing.assert_allclose(m.f0.numpy(), f0ref.numpy(), rtol=1e-6)


def test_freq_jitter_perturbs_within_range_and_keeps_nc_placeholder():
    m = pl.FrequencySoftConstraintLoss(rpm=RPM, classes=CLASSES, n_harmonics=3,
                                       freq_jitter=0.5, jitter_seed=7)
    Href, f0ref, valid = pl.build_h_matrix(RPM, CLASSES, 3)
    assert not torch.allclose(m.H, Href)
    for c in range(len(CLASSES)):
        if bool(valid[c]):
            lo, hi = 0.5 * float(f0ref[c]), 1.5 * float(f0ref[c])
            assert lo - 1e-6 <= float(m.f0[c]) <= hi + 1e-6
        else:
            # NC 的占位值必须还原为 1.0：随机化会把占位也缩放，虽被 valid 掩掉，
            # 但下游若误用 f0[NC] 会得到随机的错误量级。
            assert abs(float(m.f0[c]) - 1.0) < 1e-6
    # 谐波表与基频必须自洽：H[h, c] == (h+1) * f0[c]
    for h in range(m.H.shape[0]):
        for c in range(len(CLASSES)):
            assert abs(float(m.H[h, c]) - (h + 1) * float(m.f0[c])) < 1e-4


def test_freq_jitter_survives_set_rpm_and_is_reproducible():
    """这是本消融最容易静默失效的地方：set_rpm() 会重建 H/f0。

    若 jitter 只在外层改一次 buffer，跨工况评估切转速时随机频率会被**悄悄还原成真实频率**，
    消融就变成了"其实仍用真实先验" —— 不报错、不崩，但结论全错。
    """
    a = pl.FrequencySoftConstraintLoss(rpm=RPM, classes=CLASSES, n_harmonics=3,
                                       freq_jitter=0.5, jitter_seed=3)
    a.set_rpm(1730.0)
    # 换转速后仍是"随机频率"，而不是退回真实物理值
    Hreal_1730, _, _ = pl.build_h_matrix(1730.0, CLASSES, 3)
    assert not torch.allclose(a.H, Hreal_1730)
    # 但必须仍是 (h+1)*f0 形式且落在 ±50% 区间内（即"合法但物理错误"的表）
    _, f0real_1730, valid = pl.build_h_matrix(1730.0, CLASSES, 3)
    for c in range(len(CLASSES)):
        if bool(valid[c]):
            assert 0.5 * float(f0real_1730[c]) - 1e-6 <= float(a.f0[c])
            assert float(a.f0[c]) <= 1.5 * float(f0real_1730[c]) + 1e-6
        else:
            assert abs(float(a.f0[c]) - 1.0) < 1e-6
    # 同一 seed 的两个实例，在**同一转速**下必须得到完全相同的表（可复现）
    b = pl.FrequencySoftConstraintLoss(rpm=RPM, classes=CLASSES, n_harmonics=3,
                                       freq_jitter=0.5, jitter_seed=3)
    b.set_rpm(1730.0)
    np.testing.assert_allclose(a.H.numpy(), b.H.numpy(), rtol=1e-6)
    np.testing.assert_allclose(a.f0.numpy(), b.f0.numpy(), rtol=1e-6)


def test_freq_jitter_seed_changes_table():
    a = pl.FrequencySoftConstraintLoss(rpm=RPM, classes=CLASSES, n_harmonics=3,
                                       freq_jitter=0.5, jitter_seed=1)
    b = pl.FrequencySoftConstraintLoss(rpm=RPM, classes=CLASSES, n_harmonics=3,
                                       freq_jitter=0.5, jitter_seed=2)
    assert not torch.allclose(a.f0, b.f0)


def test_total_loss_composition(tables):
    H, f0, valid = tables
    fl = pl.FrequencySoftConstraintLoss(rpm=RPM, classes=CLASSES, n_harmonics=3)
    tl = pl.TotalLoss({'ce_weight': 1.0}, fl)
    logits = torch.randn(8, 4, requires_grad=True)
    y = torch.randint(0, 4, (8,))
    peak = torch.zeros(8, 5)
    peak[:, 0] = 110.0
    total, stats, parts = tl(logits, y, peak, lam=0.3, return_parts=True)
    manual = F.cross_entropy(logits, y) + 0.3 * fl(logits, peak)
    np.testing.assert_allclose(float(total), float(manual), rtol=1e-5)
    assert stats['lambda'] == 0.3
    assert abs(stats['loss_total'] - (stats['loss_ce'] + 0.3 * stats['loss_freq'])) < 1e-5
    assert parts['per_sample'].shape == (8,)
