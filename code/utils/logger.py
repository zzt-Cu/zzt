# -*- coding: utf-8 -*-
"""日志与结果落盘。

约定
----
- 所有实验在 ``results/logs/<run_id>/`` 下留一份完整 stdout + 环境快照；
- 所有表格写成 CSV，**带 BOM**（``utf-8-sig``），Excel 直接双击不乱码；
- 每条结果记录都带 ``run_id / seed / config_hash``，保证任何一张图都能回溯到唯一的 run；
- 绝不覆盖已有 run，避免"结果对不上但查不出原因"。
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


# ----------------------------------------------------------------------
# 路径
# ----------------------------------------------------------------------
def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def cfg_hash(cfg: Mapping[str, Any]) -> str:
    """配置指纹：任何超参改动都会改变它，用于追溯结果来源。"""
    blob = json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()[:12]


# ----------------------------------------------------------------------
# Tee 日志
# ----------------------------------------------------------------------
class Tee:
    """同时写 stdout 与文件（实验日志双写，便于事后巡查）。"""

    def __init__(self, path: str, mode: str = 'w'):
        ensure_dirs(os.path.dirname(path))
        self.path = path
        self.file = open(path, mode, encoding='utf-8', buffering=1)
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, *exc):
        sys.stdout = self.stdout
        self.file.close()
        return False


def new_run_id(prefix: str = 'run') -> str:
    return '%s_%s' % (prefix, time.strftime('%Y%m%d_%H%M%S'))


def run_log_path(log_dir: str, run_id: str) -> str:
    return os.path.join(log_dir, run_id, 'stdout.log')


# ----------------------------------------------------------------------
# 环境快照
# ----------------------------------------------------------------------
def env_snapshot() -> Dict[str, Any]:
    """记录硬件/软件环境 —— 直接供论文"可复现性声明"引用。"""
    snap: Dict[str, Any] = {
        'platform': platform.platform(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'python': sys.version.replace('\n', ' '),
        'executable': sys.executable,
        'cwd': os.getcwd(),
        'time': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    for mod in ('numpy', 'scipy', 'sklearn', 'pandas', 'matplotlib'):
        try:
            snap[mod] = __import__(mod).__version__
        except Exception:
            pass
    try:
        import torch
        snap['torch'] = torch.__version__
        snap['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            snap['gpu_name'] = torch.cuda.get_device_name(0)
            snap['gpu_capability'] = '.'.join(map(str, torch.cuda.get_device_capability(0)))
            snap['cuda_version'] = torch.version.cuda
            snap['cudnn_version'] = torch.backends.cudnn.version()
    except Exception:
        pass
    try:
        snap['nvidia_smi'] = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
             '--format=csv,noheader'],
            capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        pass
    return snap


def write_env_snapshot(out_dir: str, name: str = 'env.json') -> str:
    ensure_dirs(out_dir)
    p = os.path.join(out_dir, name)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(env_snapshot(), f, ensure_ascii=False, indent=1)
    return p


# ----------------------------------------------------------------------
# 表格写入
# ----------------------------------------------------------------------
def write_csv(path: str, rows: Sequence[Mapping[str, Any]],
              fieldnames: Optional[Sequence[str]] = None) -> str:
    """写 CSV（utf-8-sig，Excel 友好）。字段顺序默认取首行的键顺序。"""
    ensure_dirs(os.path.dirname(path))
    rows = list(rows)
    if fieldnames is None:
        if not rows:
            raise ValueError('rows 为空且未给 fieldnames，无法确定表头')
        fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(v) for k, v in r.items()})
    return path


def _fmt(v: Any) -> Any:
    if isinstance(v, float):
        return '' if v != v else ('%.6g' % v)      # NaN -> 空
    if isinstance(v, (list, tuple)):
        return ';'.join(str(x) for x in v)
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False)
    return v


def write_json(path: str, obj: Any) -> str:
    ensure_dirs(os.path.dirname(path))
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=str)
    return path


def load_json(path: str) -> Any:
    with open(path, encoding='utf-8') as f:
        return json.load(f)


# ----------------------------------------------------------------------
# 结果登记（append-only，便于跨 run 汇总）
# ----------------------------------------------------------------------
class ResultRegistry:
    """按 run 记录结果行，写到 ``results/tables/registry_<tag>.csv``。

    append-only：同一 tag 多次运行会不断追加，不会互相覆盖。
    """

    def __init__(self, tables_dir: str, tag: str = 'main'):
        ensure_dirs(tables_dir)
        self.path = os.path.join(tables_dir, 'registry_%s.csv' % tag)
        self.tag = tag

    def append(self, rows: Sequence[Mapping[str, Any]]) -> str:
        rows = list(rows)
        if not rows:
            return self.path
        exists = os.path.exists(self.path) and os.path.getsize(self.path) > 0
        if exists:
            with open(self.path, 'r', encoding='utf-8-sig', newline='') as f:
                header = next(csv.reader(f), [])
        else:
            header = list(rows[0].keys())
        for r in rows:
            for k in r:
                if k not in header:
                    header.append(k)
        # 已有表头可能缺字段，重写一次保证列齐全
        if exists:
            with open(self.path, 'r', encoding='utf-8-sig', newline='') as f:
                old = list(csv.DictReader(f))
            with open(self.path, 'w', newline='', encoding='utf-8-sig') as f:
                w = csv.DictWriter(f, fieldnames=header, extrasaction='ignore')
                w.writeheader()
                for r in old:
                    w.writerow(r)
                for r in rows:
                    w.writerow({k: _fmt(v) for k, v in r.items()})
        else:
            with open(self.path, 'w', newline='', encoding='utf-8-sig') as f:
                w = csv.DictWriter(f, fieldnames=header, extrasaction='ignore')
                w.writeheader()
                for r in rows:
                    w.writerow({k: _fmt(v) for k, v in r.items()})
        return self.path


if __name__ == '__main__':
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), 'results')
    h = cfg_hash({'a': 1, 'b': [1, 2]})
    print('cfg_hash demo:', h, '(应稳定)')
    p = write_env_snapshot(os.path.join(base, 'logs', 'demo'))
    env = load_json(p)
    print('env keys:', sorted(env.keys()))
    p2 = write_csv(os.path.join(base, 'tables', '_demo.csv'),
                   [{'method': 'CNN-CE', 'acc': 0.8123, 'note': None},
                    {'method': 'CNN-Freq', 'acc': 0.9056, 'note': float('nan')}])
    print('csv written:', p2)
    print('--- 内容 ---')
    print(open(p2, encoding='utf-8-sig').read())
