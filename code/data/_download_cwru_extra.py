# -*- coding: utf-8 -*-
"""下载 CWRU **尺寸细分**数据（0.014" / 0.021"），用于 severity-shift 实验。

编号来源：2026-09-22 从 CWRU 官网 12k Drive End 页面逐条核实（不是凭记忆）：
https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data

    0.014"  IR014_0..3 = 169,170,171,172
            B014_0..3  = 185,186,187,188
            OR014@6_0..3 = 197,198,199,200      (OR014 仅 @6:00 有数据)
    0.021"  IR021_0..3 = 209,210,211,212
            B021_0..3  = 222,223,224,225
            OR021@6_0..3 = 234,235,236,237      ← 0hp 是 234，不是 235！
            (OR021 另有 @3/@12，但为与 0.007"(@6) 可比，这里只取 @6:00)

为什么做尺寸细分：BPFO/BPFI/BSF **只取决于轴承几何与转速，与缺陷尺寸无关**；
尺寸改变的是特征频率处的**幅值**。所以"幅值型物理量"在 severity shift 下
理论上比"峰位型"更占优 —— 这正是要验证的假设。

用法：
    python _download_cwru_extra.py            # 下载 + 校验
    python _download_cwru_extra.py --check     # 只校验本地已有文件
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import ssl
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

RAW = os.environ.get('CWRU_RAW') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data', 'raw')
URL = 'https://engineering.case.edu/sites/default/files/%d.mat'
WORKERS = 6

EXTRA = {
    'IR014': {0: 169, 1: 170, 2: 171, 3: 172},
    'B014':  {0: 185, 1: 186, 2: 187, 3: 188},
    'OR014': {0: 197, 1: 198, 2: 199, 3: 200},
    'IR021': {0: 209, 1: 210, 2: 211, 3: 212},
    'B021':  {0: 222, 1: 223, 2: 224, 3: 225},
    'OR021': {0: 234, 1: 235, 2: 236, 3: 237},
}

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
UA = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}


def fetch_one(code: int, tries: int = 4):
    dest = os.path.join(RAW, '%d.mat' % code)
    if os.path.exists(dest) and os.path.getsize(dest) > 100000:
        return code, os.path.getsize(dest), 'cached'
    url = URL % code
    last = None
    for k in range(1, tries + 1):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=180, context=ctx) as r:
                data = r.read()
            if len(data) < 100000:
                raise IOError('too small %d' % len(data))
            tmp = dest + '.part'
            with open(tmp, 'wb') as f:
                f.write(data)
            os.replace(tmp, dest)
            return code, len(data), 'downloaded'
        except Exception as e:                                   # noqa: BLE001
            last = '%s: %s' % (type(e).__name__, e)
            time.sleep(2 * k)
    return code, None, 'FAIL ' + str(last)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='只校验，不下载')
    args = ap.parse_args(argv)

    os.makedirs(RAW, exist_ok=True)
    jobs = [(cls, hp, code) for cls, m in EXTRA.items() for hp, code in sorted(m.items())]
    print('本次清单 %d 个文件（尺寸细分）' % len(jobs))

    t0 = time.time()
    if not args.check:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = {ex.submit(fetch_one, c): (cls, hp, c) for cls, hp, c in jobs}
            for fut in as_completed(futs):
                cls, hp, c = futs[fut]
                try:
                    code, size, status = fut.result()
                except Exception as e:                           # noqa: BLE001
                    code, size, status = c, None, 'EXC %s' % e
                print('[%5.1fs] %-6s %dhp %4d.mat  %-11s %s'
                      % (time.time() - t0, cls, hp, code, status,
                         ('%.2f MB' % (size / 1e6)) if size else ''))

    from scipy.io import loadmat
    report, bad = [], 0
    print('\n=== 校验 ===')
    for cls, hp, code in jobs:
        p = os.path.join(RAW, '%d.mat' % code)
        rec = dict(cls=cls, hp=hp, code=code)
        if not os.path.exists(p) or os.path.getsize(p) < 100000:
            rec['status'] = 'MISSING'
            bad += 1
            report.append(rec)
            print('%-6s %dhp %4d.mat  MISSING' % (cls, hp, code))
            continue
        rec['sha256_16'] = hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]
        try:
            md = loadmat(p)
            keys = sorted(k for k in md if k.endswith('DE_time'))
            want = 'X%03d_DE_time' % code
            rec['all_de_keys'] = keys
            rec['de_key_used'] = want
            if want in md:
                arr = np.asarray(md[want]).ravel().astype(np.float64)
                rec['n_samples'] = int(arr.size)
                rec['rms'] = round(float(np.sqrt((arr ** 2).mean())), 4)
                rec['kurt'] = round(float(((arr - arr.mean()) ** 4).mean()
                                          / (arr.var() ** 2 + 1e-30)), 3)
                extra = [k for k in keys if k != want]
                if extra:
                    rec['extra_de_keys'] = extra
                rec['status'] = 'verified'
                print('%-6s %dhp %4d.mat  ok  n=%-7d rms=%-8.4f kurt=%-7.3f %s'
                      % (cls, hp, code, rec['n_samples'], rec['rms'], rec['kurt'],
                         ('extra=%s' % extra) if extra else ''))
            else:
                rec['status'] = 'MISSING_DE_VAR'
                bad += 1
                print('%-6s %dhp %4d.mat  MISSING_DE_VAR (found %s)' % (cls, hp, code, keys))
        except Exception as e:                                   # noqa: BLE001
            rec['status'] = 'LOADMAT_FAIL:%s' % type(e).__name__
            bad += 1
            print('%-6s %dhp %4d.mat  %s' % (cls, hp, code, rec['status']))
        report.append(rec)

    out = os.path.join(RAW, 'manifest_extra.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print('\nBAD_COUNT=%d  ->  %s  (%.1fs)' % (bad, out, time.time() - t0))
    print('提示：kurt（峭度）可用于粗查"缺陷是否真的更大"——0.021" 通常峭度更高，')
    print('      但它不是判据，仅作合理性交叉参考。')


if __name__ == '__main__':
    main()
