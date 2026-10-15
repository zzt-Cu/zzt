# -*- coding: utf-8 -*-
"""CWRU 12kHz 驱动端数据并发下载 + 校验。

URL 模式（2026-09-22 实测可用）: https://engineering.case.edu/sites/default/files/{code}.mat
单连接实测只有 ~28 KB/s，故改为多线程并发；每个文件独立重试。
"""
import os, sys, json, time, hashlib, ssl, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

RAW = os.environ.get('CWRU_RAW') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data', 'raw')
LOG = os.path.join(RAW, '_download.log')
URL = 'https://engineering.case.edu/sites/default/files/%d.mat'
WORKERS = 6

MANIFEST = {
    'NC':    {0: 97,  1: 98,  2: 99,  3: 100},
    'IR007': {0: 105, 1: 106, 2: 107, 3: 108},
    'B007':  {0: 118, 1: 119, 2: 120, 3: 121},
    'OR007': {0: 130, 1: 131, 2: 132, 3: 133},   # OR007@6 中心位置
}

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
UA = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

_logf = open(LOG, 'w', encoding='utf-8')


def log(msg):
    print(msg)
    _logf.write(msg + '\n')
    _logf.flush()


def fetch_one(code, tries=4):
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
        except Exception as e:
            last = '%s: %s' % (type(e).__name__, e)
            time.sleep(2 * k)
    return code, None, 'FAIL ' + str(last)


def main():
    os.makedirs(RAW, exist_ok=True)
    jobs = [(cls, hp, code) for cls, m in MANIFEST.items() for hp, code in sorted(m.items())]
    log('开始下载 %d 个文件，并发 %d' % (len(jobs), WORKERS))
    t0 = time.time()
    results = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(fetch_one, c): (cls, hp, c) for cls, hp, c in jobs}
        for fut in as_completed(futs):
            cls, hp, c = futs[fut]
            try:
                code, size, status = fut.result()
            except Exception as e:
                code, size, status = c, None, 'EXC %s' % e
            results[code] = (cls, hp, size, status)
            log('[%5.1fs] %-6s %dhp %4d.mat  %-11s %s' % (
                time.time() - t0, cls, hp, code, status,
                ('%.2f MB' % (size / 1e6)) if size else ''))

    # 校验
    from scipy.io import loadmat
    report = []
    for cls, hp, code in jobs:
        p = os.path.join(RAW, '%d.mat' % code)
        rec = dict(cls=cls, hp=hp, code=code)
        if not os.path.exists(p) or os.path.getsize(p) < 100000:
            rec['status'] = 'MISSING'
            report.append(rec)
            continue
        rec['size'] = os.path.getsize(p)
        rec['sha256_16'] = hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]
        try:
            md = loadmat(p)
            keys = sorted(k for k in md if k.endswith('DE_time'))
            rec['all_de_keys'] = keys
            # ⚠️ 必须按官方编号取 DE 变量，不能取 keys[0]。
            #    实测 99.mat 内同时含 X098_DE_time(1hp) 与 X099_DE_time(2hp)，
            #    取第一个会静默把 2hp 样本换成 1hp，跨工况实验直接失效。
            want = 'X%03d_DE_time' % code
            rec['de_key_used'] = want
            if want in md:
                if len(keys) > 1:
                    rec['extra_de_keys'] = [k for k in keys if k != want]
                arr = np.asarray(md[want]).ravel().astype(np.float64)
                rec['n_samples'] = int(arr.size)
                rec['rms'] = round(float(np.sqrt((arr ** 2).mean())), 4)
                rec['peak'] = round(float(np.abs(arr).max()), 3)
                rec['status'] = 'verified'
            else:
                rec['status'] = 'MISSING_DE_VAR:%s (found %s)' % (want, keys)
        except Exception as e:
            rec['status'] = 'LOADMAT_FAIL:%s' % type(e).__name__
        report.append(rec)

    with open(os.path.join(RAW, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=1)

    log('\n=== SUMMARY (%.1fs) ===' % (time.time() - t0))
    for r in report:
        log('%-6s %dhp %4d.mat  %-12s n=%-7s rms=%-9s %s' % (
            r['cls'], r['hp'], r['code'], r['status'],
            r.get('n_samples', '-'), r.get('rms', '-'),
            ','.join(r.get('all_de_keys', []))))
    bad = [r for r in report if r['status'] != 'verified']
    log('BAD_COUNT=%d' % len(bad))
    for r in bad:
        log('BAD: %s' % json.dumps(r, ensure_ascii=False))
    extra = [r for r in report if r.get('extra_de_keys')]
    log('FILES_WITH_EXTRA_DE_KEYS=%d' % len(extra))
    for r in extra:
        log('  !! %d.mat 额外含 %s —— 若按变量名顺序取值会静默取错工况'
            % (r['code'], r['extra_de_keys']))
    _logf.close()


main()
