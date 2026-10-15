# -*- coding: utf-8 -*-
"""数据层测试：官方编号取值、切窗/划分无泄漏、包络谱、加噪 SNR、清单核对。

这里每一条测试都对应一个"不报错但会让结论悄悄错掉"的风险点，
不是形式主义的覆盖率填空。
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

CODE = Path(__file__).resolve().parents[1]
ROOT = CODE.parent
sys.path.insert(0, str(CODE))

from data import bearing_freq as bf                       # noqa: E402
from data import cwru_loader as cl                        # noqa: E402
from data import envelope_spectrum as es                  # noqa: E402

RAW = ROOT / 'data' / 'raw'
HAS_DATA = (RAW / '99.mat').exists()
need_data = pytest.mark.skipif(not HAS_DATA, reason='CWRU 原始数据未下载')


# ----------------------------------------------------------------------
# 官方编号与文件名
# ----------------------------------------------------------------------
def test_de_var_name_format():
    assert cl.de_var_name(97) == 'X097_DE_time'
    assert cl.de_var_name(105) == 'X105_DE_time'
    assert cl.de_var_name(7) == 'X007_DE_time'


def test_bare_name_does_not_truncate_on_inner_dot():
    """坑③：带小数点的规范化文件名用 .stem 会被截断。"""
    assert cl.bare_name('CWRU_IR_0.007in_0HP_1797rpm_official105.mat') \
        == 'CWRU_IR_0.007in_0HP_1797rpm_official105'
    assert cl.bare_name(r'D:\a\b\0.007_x.mat') == '0.007_x'
    assert cl.bare_name('plain') == 'plain'
    # 反例：真正让 Path.stem 截断的是"文件名本身不带 .mat"的情形 ——
    # 此时 Path 会把 '.007in' 整个当成扩展名，stem 只剩前半段。
    assert Path('CWRU_IR_0.007in').stem == 'CWRU_IR_0'
    assert cl.bare_name('CWRU_IR_0.007in') == 'CWRU_IR_0.007in'   # 本项目实现不受影响
    # 对照组：带 .mat 时 Path.stem 恰好是对的（所以这个坑更隐蔽 —— 只在无后缀时不成立）
    assert Path('CWRU_IR_0.007in_0HP.mat').stem == 'CWRU_IR_0.007in_0HP'


def test_code_to_class_load_roundtrip():
    for cls, m in cl.FILE_CODES.items():
        for hp, code in m.items():
            assert cl.code_to_class_load(code) == (cls, hp)
    with pytest.raises(KeyError):
        cl.code_to_class_load(9999)


@need_data
def test_99mat_takes_official_variable_not_first_de_key():
    """坑①：99.mat 内同时含 X098_DE_time(1HP) 与 X099_DE_time(2HP)。

    按"第一个以 _DE_time 结尾的变量"取值会静默把 2HP 换成 1HP，
    跨工况实验会退化成同工况。本测试直接守住这一点。
    """
    from scipy.io import loadmat
    mat = loadmat(str(RAW / '99.mat'))
    keys = sorted(k for k in mat if k.endswith('DE_time'))
    assert 'X098_DE_time' in keys and 'X099_DE_time' in keys, '前提变了，请复核'
    assert keys[0] == 'X098_DE_time', '排序第一个正是 1HP 的数据（陷阱）'

    sig, rec = cl.load_de_signal(99, str(RAW))
    assert rec.de_key == 'X099_DE_time'
    assert rec.extra_de_keys == ['X098_DE_time']
    assert rec.warnings, '出现额外 DE 变量时必须留下告警'
    # 取到的必须是 2HP 的那条，而不是 1HP
    ref = np.asarray(mat['X099_DE_time']).squeeze().astype(np.float64)
    np.testing.assert_allclose(sig, ref)
    # 与 1HP 版本必须不同。注意：两者长度不等（485063 vs 483903），
    # 直接 np.allclose 会因广播失败而报 ValueError，所以先比长度、再比公共前缀。
    other = np.asarray(mat['X098_DE_time']).squeeze().astype(np.float64)
    assert sig.size != other.size, '取错了变量：长度与 1HP 版本相同'
    m = min(sig.size, other.size)
    assert not np.allclose(sig[:m], other[:m])


@need_data
def test_99mat_2hp_differs_from_98mat_1hp():
    """上一条的推论：99.mat 取对了就不该等于 98.mat。"""
    a, _ = cl.load_de_signal(99, str(RAW))
    b, _ = cl.load_de_signal(98, str(RAW))
    n = min(a.size, b.size)
    assert not np.allclose(a[:n], b[:n])


@need_data
def test_raw_manifest_is_consistent():
    import yaml
    with open(CODE / 'configs' / 'default.yaml', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg['data']['raw_dir'] = str(RAW)
    recs = cl.build_raw_manifest(cfg, verbose=False)
    # 每个类别 × 4 个负载；类别数随 FILE_CODES 增长（现含 0.007/0.014/0.021 三档尺寸）
    assert len(recs) == len(cl.FILE_CODES) * 4, len(recs)
    assert all(r.path for r in recs), '有文件没读到'
    assert all(r.n_samples > 100000 for r in recs)
    assert all(r.de_key == cl.de_var_name(r.code) for r in recs)
    # 只有 99.mat 会带额外 DE 变量
    extra = {r.code: r.extra_de_keys for r in recs if r.extra_de_keys}
    assert extra == {99: ['X098_DE_time']}, '额外 DE 变量集合变了：%s' % extra


# ----------------------------------------------------------------------
# 切窗与划分：泄漏是这里唯一要防的事
# ----------------------------------------------------------------------
def test_cap_windows_is_equal_interval():
    X = np.arange(1000).reshape(100, 10).astype(np.float32)
    Y, kept = cl.cap_windows(X, 10)
    assert Y.shape == (10, 10)
    assert kept[0] == 0 and kept[-1] == 99                     # 覆盖到末尾，不是只取前 10 个
    diffs = np.diff(kept)
    assert diffs.max() - diffs.min() <= 1, '应等间隔'


def test_split_blocks_is_mutually_exclusive_and_complete():
    n = 1001
    blocks = cl.split_blocks(n, (0.6, 0.2, 0.2))
    assert blocks[0][0] == 0 and blocks[-1][1] == n
    for (s1, e1), (s2, e2) in zip(blocks, blocks[1:]):
        assert e1 == s2, '块之间必须首尾相接，不能有缝也不能重叠'
    assert sum(e - s for s, e in blocks) == n


def test_split_blocks_rejects_bad_fractions():
    with pytest.raises(ValueError):
        cl.split_blocks(100, (0.5, 0.2, 0.2))


def test_no_window_overlap_between_splits():
    """按时间块切分后，train/val/test 的窗口区间必须两两不相交。"""
    n = 24000
    window, stride = 2048, 512
    spans = []
    for s, e in cl.split_blocks(n, (0.6, 0.2, 0.2)):
        seg = np.arange(n)[s:e]
        starts = np.arange(0, seg.size - window + 1, stride)
        assert starts.size > 0, '块太小，切不出窗口'
        spans.append(list(zip((starts + s).tolist(), (starts + s + window).tolist())))
    # 每对划分之间检查区间不相交
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            for a_s, a_e in spans[i]:
                for b_s, b_e in spans[j]:
                    assert a_e <= b_s or b_e <= a_s, \
                        '划分 %d 的窗口 [%d,%d) 与划分 %d 的 [%d,%d) 重叠' \
                        % (i, a_s, a_e, j, b_s, b_e)


def test_zscore_is_per_window_only():
    """标准化统计量只能来自窗口自身：改动别的行不能影响本行结果。"""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((5, 64)).astype(np.float32)
    Z1 = cl.zscore_windows(X)
    X2 = X.copy()
    X2[0] *= 1000.0 + 7.0                                      # 大幅改动第 0 行
    Z2 = cl.zscore_windows(X2)
    np.testing.assert_allclose(Z1[1:], Z2[1:], atol=1e-6)
    np.testing.assert_allclose(Z1.mean(axis=1), 0.0, atol=1e-5)
    np.testing.assert_allclose(Z1.std(axis=1), 1.0, atol=1e-4)


def test_segment_shape_and_values():
    sig = np.arange(1000, dtype=np.float64)
    X = cl.segment(sig, 100, 50)
    assert X.shape == (19, 100)
    np.testing.assert_allclose(X[0], sig[:100])
    np.testing.assert_allclose(X[1], sig[50:150])
    assert cl.segment(sig, 2000, 100).shape == (0, 2000)


# ----------------------------------------------------------------------
# 加噪：必须在时域加，且能反算真实 SNR
# ----------------------------------------------------------------------
@pytest.mark.parametrize('snr_db', [0.0, 6.0, 12.0, -3.0])
def test_add_noise_snr_roundtrip(snr_db):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 512)).astype(np.float32)
    Xn = cl.add_noise_at_snr(X, snr_db, rng)
    got = cl.measure_snr(X, Xn)
    assert abs(got - snr_db) < 1.0, '请求 %.1f dB 实测 %.2f dB' % (snr_db, got)


def test_add_noise_rejects_envelope_spectrum_usage():
    """文档化约束：加噪必须在波形上做（包络谱为非负量，叠加高斯噪声会造出负谱线）。"""
    rng = np.random.default_rng(0)
    env_like = np.abs(rng.standard_normal((4, 256))) + 1e-3
    noisy = cl.add_noise_at_snr(env_like, 0.0, rng)
    assert (noisy < 0).any(), '这正是"在包络谱上加噪"会产生负谱线的反例证据'


# ----------------------------------------------------------------------
# 包络谱
# ----------------------------------------------------------------------
def test_envelope_spectrum_finds_modulation_frequency():
    fs, n, f_mod, f_car = 12000, 2048, 107.37, 3000.0
    t = np.arange(n) / fs
    rng = np.random.default_rng(0)
    sig = (1.0 + 0.8 * np.cos(2 * np.pi * f_mod * t)) * np.cos(2 * np.pi * f_car * t)
    sig += 0.05 * rng.standard_normal(n)
    cfg = es.EnvelopeConfig(fs=fs, band=(500, 5000), n_fft=8192, n_peaks=5)
    err = es.nearest_harmonic_error(es.peak_frequencies(sig, cfg), f_mod, 3)
    assert err < 0.02, '相对误差 %.4f 过大' % err


def test_envelope_spectrum_handles_multiple_harmonics():
    fs, n, f0 = 12000, 4096, 107.37
    t = np.arange(n) / fs
    carrier = np.cos(2 * np.pi * 3000 * t)
    sig = (1.0 + 0.6 * np.cos(2 * np.pi * f0 * t)
           + 0.3 * np.cos(2 * np.pi * 2 * f0 * t)) * carrier
    cfg = es.EnvelopeConfig(fs=fs, band=(500, 5000), n_fft=16384, n_peaks=5)
    pk = es.peak_frequencies(sig, cfg)
    ratios = pk / f0
    hits = np.min(np.abs(ratios[:, None] - np.array([1.0, 2.0, 3.0])[None, :]), axis=1)
    assert hits.min() < 0.03, '未能匹配到 f0 或其谐波: %s' % np.round(ratios, 3)


def test_bandpass_rejects_illegal_band():
    cfg = es.EnvelopeConfig(fs=12000, band=(500, 5000))
    with pytest.raises(ValueError):
        es.bandpass(np.zeros(1024), cfg.fs, (500, 7000), cfg.order)   # 超过 Nyquist


def test_peak_frequencies_pads_to_k():
    cfg = es.EnvelopeConfig(fs=12000, n_fft=2048, n_peaks=5)
    pk = es.peak_frequencies(np.zeros(2048), cfg)
    assert pk.shape == (5,)


# ----------------------------------------------------------------------
# 轴承特征频率（几何自校验）
# ----------------------------------------------------------------------
def test_bearing_geometry_matches_cwru_official_multipliers():
    """本模块实现必须与 CWRU 官方倍数表吻合（BSF 用冲击率口径 = 2×公式值）。"""
    err = bf.selfcheck()
    assert max(err.values()) < 0.5, err


def test_bsf_2x_is_exactly_twice_bsf():
    for rpm in (1730, 1750, 1772, 1797):
        fr = bf.shaft_freq(rpm)
        assert abs(bf.bsf_2x(fr) - 2.0 * bf.bsf(fr)) < 1e-12


def test_class_frequency_specs_nc_has_no_frequency():
    specs = bf.class_frequency_specs(1797.0, ['NC', 'IR007', 'OR007', 'B007'], 3)
    assert specs['NC'].fundamental is None and specs['NC'].harmonics == []
    assert specs['IR007'].fault_type == 'BPFI'
    assert specs['OR007'].fault_type == 'BPFO'
    assert specs['B007'].fault_type == 'BSF'
    np.testing.assert_allclose(specs['IR007'].harmonics,
                               [specs['IR007'].fundamental * h for h in (1, 2, 3)])


def test_theoretical_frequency_matrix_mask():
    M, mask = bf.theoretical_frequency_matrix(1797.0, ['NC', 'IR007', 'OR007', 'B007'], 3)
    assert len(M) == 3 and len(M[0]) == 4
    assert mask == [False, True, True, True]
    assert all(M[h][0] == 0.0 for h in range(3)), 'NC 列应全为 0 占位'


def test_fault_frequency_shifts_with_load():
    """转速下降 -> 特征频率同比例下降（跨工况的物理前提）。"""
    a = bf.fault_frequencies(1797.0)['BPFO']
    b = bf.fault_frequencies(1730.0)['BPFO']
    assert a > b
    assert abs(a / b - 1797.0 / 1730.0) < 1e-6


# ----------------------------------------------------------------------
# 动态类别数（evaluate.py --classes）
# 标签 = 类别列表下标 ⇒ 换集合就是换标签空间，缓存与 way 数都必须跟着变。
# 这类错**不报错**，只会让"3 类实验"悄悄跑在"4 类池"上，所以专门守住。
# ----------------------------------------------------------------------
def test_label_map_follows_given_classes():
    assert cl.label_map(['NC', 'IR007', 'OR007']) == {'NC': 0, 'IR007': 1, 'OR007': 2}
    assert cl.label_map() == {c: i for i, c in enumerate(cl.CLASS_ORDER)}
    # B007 在默认序里是 3；3 类配置下它压根不该出现在标签空间
    assert 'B007' not in cl.label_map(['NC', 'IR007', 'OR007'])


def test_cache_path_fingerprints_classes():
    base = {'data': {'window': 2048, 'stride_train': 512, 'stride_eval': 512,
                     'max_windows_per_file': 400, 'balance_pools': True,
                     'processed_dir': 'x',
                     'classes': ['NC', 'IR007', 'OR007', 'B007']}}
    p4 = cl.cache_path(base)
    assert p4 == cl.cache_path(base), '同输入必须给同缓存名'
    c3 = {'data': dict(base['data'], classes=['NC', 'IR007', 'OR007'])}
    p3 = cl.cache_path(c3)
    assert p4 != p3, '类别集合不同却共用缓存 —— 3 类实验会悄悄跑在 4 类池上'
    # 顺序不同 = 标签空间不同，指纹也必须不同
    c3r = {'data': dict(base['data'], classes=['IR007', 'NC', 'OR007'])}
    assert cl.cache_path(c3r) != p3


@need_data
def test_build_pools_respects_class_subset():
    """3 类配置下池里只能出现标签 0..2，且 n_per_class 的键就是这 3 类。"""
    import yaml
    with open(CODE / 'configs' / 'default.yaml', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg['data']['raw_dir'] = str(RAW)
    cfg['data']['classes'] = ['NC', 'IR007', 'OR007']
    cfg['data']['max_windows_per_file'] = 40         # 只为跑通，不求精度
    cfg['freq']['shaft_rpm'] = {int(k): float(v) for k, v in cfg['freq']['shaft_rpm'].items()}
    pools = cl.build_pools(cfg, verbose=False)
    assert {'train', 'val', 'test'} <= set(pools)
    want = {'NC', 'IR007', 'OR007'}
    for name, wp in pools.items():
        assert set(wp.n_per_class) == want, (name, wp.n_per_class)
        assert wp.y.max() <= 2, '%s 出现越界标签 %d（说明用错了默认 LABEL_MAP）' % (name, wp.y.max())
        assert wp.y.min() >= 0
    # 默认 4 类映射里 B007=3；3 类下标签 3 必须完全不存在
    assert pools['train'].y.max() == 2
