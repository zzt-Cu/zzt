# -*- coding: utf-8 -*-
"""模型层测试：参数量硬约束、形状、embedding、复杂度估算、可复现初始化。

参数量 < 50K 是本项目的**硬叙事前提**（边缘部署 + "收益不是靠堆参数换来"），
所以这里用断言把它焊死，任何结构改动导致超限都会立刻失败。
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from models import bearing_cnn as bc                        # noqa: E402


# ----------------------------------------------------------------------
# 参数量硬约束
# ----------------------------------------------------------------------
def test_default_model_under_50k():
    m = bc.BearingCNN()
    n = bc.count_parameters(m)
    assert n < 50000, '参数量 %d 超限' % n
    assert n == 45732, '参数量变了(%d)，需同步更新论文里的数字与 INT8 体积估算' % n


def test_config_match_config_yaml():
    """YAML 里的配置也必须满足 < 50K（配置与代码默认值不能各说各话）。"""
    import yaml
    with open(CODE / 'configs' / 'default.yaml', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    m = bc.BearingCNN.from_cfg(cfg)
    n = bc.count_parameters(m)
    assert n < int(cfg['model']['max_params']), '%d >= %d' % (n, cfg['model']['max_params'])
    assert n == 45732


def test_assert_param_budget_raises_when_over():
    m = bc.BearingCNN(channels=(64, 128, 256, 512))
    with pytest.raises(AssertionError):
        bc.assert_param_budget(m, 50000)


def test_parameter_breakdown_sums_to_total():
    m = bc.BearingCNN()
    bd = bc.parameter_breakdown(m)
    assert sum(v for k, v in bd.items() if k != 'TOTAL') == bd['TOTAL']


def test_exactly_four_conv_and_two_linear_layers():
    """文档规定：4 层卷积 + 2 层全连接。"""
    m = bc.BearingCNN()
    convs = [mod for mod in m.modules() if isinstance(mod, torch.nn.Conv1d)]
    lins = [mod for mod in m.modules() if isinstance(mod, torch.nn.Linear)]
    assert len(convs) == 4
    assert len(lins) == 2


def test_rejects_wrong_layer_count():
    with pytest.raises(AssertionError):
        bc.BearingCNN(channels=(16, 32, 64))


# ----------------------------------------------------------------------
# 形状与接口
# ----------------------------------------------------------------------
@pytest.mark.parametrize('length', [1024, 2048, 4096])
def test_forward_shapes(length):
    m = bc.BearingCNN()
    x = torch.randn(6, 1, length)
    assert m(x).shape == (6, 4)
    e = m(x, return_embedding=True)
    assert e.shape == (6, m.embed_dim)


def test_accepts_2d_input():
    m = bc.BearingCNN()
    assert m(torch.randn(5, 2048)).shape == (5, 4)


def test_embedding_is_post_relu_nonneg():
    m = bc.BearingCNN()
    e = m.embed(torch.randn(4, 1, 2048))
    assert (e >= 0).all(), 'embed 是 ReLU 之后，应非负'


def test_pool_modes_shapes():
    for mode, mult in [('gap_gmp', 2), ('gap', 1), ('gmp', 1)]:
        m = bc.BearingCNN(pool_mode=mode)
        assert m.feat_dim == m.out_channels * mult
        assert m(torch.randn(2, 1, 2048)).shape == (2, 4)
    with pytest.raises(AssertionError):
        bc.BearingCNN(pool_mode='nope')


def test_n_way_is_configurable():
    m = bc.BearingCNN(n_way=3)
    assert m(torch.randn(2, 1, 2048)).shape == (2, 3)


# ----------------------------------------------------------------------
# 复杂度估算
# ----------------------------------------------------------------------
def test_flops_estimate_is_positive_and_scales_with_length():
    m = bc.BearingCNN()
    f1 = bc.estimate_flops(m, 1024)
    f2 = bc.estimate_flops(m, 2048)
    assert f1['flops'] > 0 and f2['flops'] > f1['flops']
    assert f2['flops'] == pytest.approx(2.0 * f2['macs'])
    # 单样本 FLOPs 应在 M 量级（与 45K 参数的小模型相符）
    assert 0.5e6 < f2['flops'] < 20e6


def test_int8_weight_size_estimate():
    n = bc.count_parameters(bc.BearingCNN())
    kb = n / 1024.0
    assert kb < 100, 'INT8 权重量 %.1f KB 应小于 100KB 的部署预算' % kb


# ----------------------------------------------------------------------
# 可复现初始化
# ----------------------------------------------------------------------
def test_seeded_init_is_reproducible():
    torch.manual_seed(123)
    a = bc.BearingCNN()
    torch.manual_seed(123)
    b = bc.BearingCNN()
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)


def test_different_seeds_give_different_init():
    torch.manual_seed(1)
    a = bc.BearingCNN()
    torch.manual_seed(2)
    b = bc.BearingCNN()
    assert not torch.equal(next(a.parameters()), next(b.parameters()))


def test_lr_scheduler_and_optimizer_accept_model():
    m = bc.BearingCNN()
    opt = torch.optim.AdamW(m.parameters(), lr=2e-4, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
    x = torch.randn(8, 1, 2048)
    y = torch.randint(0, 4, (8,))
    loss = torch.nn.functional.cross_entropy(m(x), y)
    loss.backward()
    opt.step()
    sch.step()
    assert opt.param_groups[0]['lr'] < 2e-4, 'Cosine 应该让 lr 下降'


def test_gradients_finite_on_real_shaped_batch():
    torch.manual_seed(0)
    m = bc.BearingCNN()
    x = torch.randn(64, 1, 2048)
    y = torch.randint(0, 4, (64,))
    loss = torch.nn.functional.cross_entropy(m(x), y)
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
