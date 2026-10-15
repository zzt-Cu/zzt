# -*- coding: utf-8 -*-
"""随机性控制：固定随机种子，保证 episode 采样、权重初始化、数据划分可复现。

可复现性要求（对应论文"可复现性声明"）：
- Python ``random`` / NumPy 全局 / PyTorch CPU & CUDA 全部固定；
- ``torch.backends.cudnn.deterministic = True``；
- DataLoader 的 worker 种子由 ``worker_init_fn`` 派生自同一个 base seed；
- 每个方法 × 每种配置 × 每个重复次数使用 ``base_seed + 偏移`` 的**独立**种子，
  避免不同 run 之间共享同一串随机数。
"""
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def setup_seed(seed: int, deterministic: bool = True, verbose: bool = False) -> int:
    """固定全部随机源，返回实际使用的种子。"""
    seed = int(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # 部分算子（如某些 reduce）在严格确定模式下需要该开关
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
    except ImportError:
        pass
    if verbose:
        print('[seed] base seed = %d (deterministic=%s)' % (seed, deterministic))
    return seed


def derive_seed(base_seed: int, *tags) -> int:
    """由 base seed 与一组标签派生稳定子种子。

    使用位置无关的稳定哈希（不用 Python 内置 ``hash``，它带随机盐），
    保证同一 (base_seed, tags) 在任何进程/机器上得到同一个值。
    """
    import hashlib
    payload = '%d|%s' % (int(base_seed), '|'.join(str(t) for t in tags))
    h = hashlib.sha256(payload.encode('utf-8')).hexdigest()
    return int(h[:8], 16) % (2 ** 31 - 1)


def rng_for(base_seed: int, *tags) -> np.random.Generator:
    """取得该标签组合专属的 NumPy 随机数发生器（互不干扰）。"""
    return np.random.default_rng(derive_seed(base_seed, *tags))


def worker_init_fn_from(base_seed: int):
    """DataLoader worker 初始化函数工厂。"""
    def _fn(worker_id: int):
        s = derive_seed(base_seed, 'worker', worker_id)
        np.random.seed(s)
        random.seed(s)
    return _fn


def describe() -> dict:
    """记录当前随机性相关环境，供日志/复现声明使用。"""
    info = {'pythonhashseed': os.environ.get('PYTHONHASHSEED')}
    try:
        import torch
        info['torch'] = torch.__version__
        info['cudnn_deterministic'] = torch.backends.cudnn.deterministic
        info['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info['gpu'] = torch.cuda.get_device_name(0)
            info['cuda_version'] = torch.version.cuda
    except ImportError:
        pass
    info['numpy'] = np.__version__
    return info


if __name__ == '__main__':
    s = setup_seed(42, verbose=True)
    print('derive_seed(42,"a")   =', derive_seed(42, 'a'))
    print('derive_seed(42,"a")   =', derive_seed(42, 'a'), '(应稳定一致)')
    print('derive_seed(42,"b")   =', derive_seed(42, 'b'), '(应不同)')
    print('derive_seed(43,"a")   =', derive_seed(43, 'a'), '(应不同)')
    g1, g2 = rng_for(42, 'x'), rng_for(42, 'y')
    a1, a2 = g1.random(3), g2.random(3)
    g1b = rng_for(42, 'x')
    assert np.allclose(g1b.random(3), a1), 'rng_for 不可复现'
    assert not np.allclose(a1, a2), '不同标签的 rng 不应相同'
    print('\n[OK] rng_for 可复现且按标签隔离')
    print('环境:', describe())
