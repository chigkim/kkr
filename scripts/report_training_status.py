#!/usr/bin/env python3
'''Report Kokoro/StyleTTS2 training progress and macOS memory status.

This script is intentionally read-only. It parses the train log, finds the active
train process, and prints the memory stats that are useful for Apple Silicon/MPS
runs.
'''

from __future__ import annotations

import argparse
import re
import subprocess
from collections import defaultdict
from pathlib import Path


STEP_RE = re.compile(
    r'(?P<time>\d\d:\d\d:\d\d).*?'
    r'Epoch(?: \[|=)(?P<epoch>\d+)/(?:\d+).*?'
    r'Step(?: \[|=)(?P<step>\d+)/(?:\d+).*?'
    r'LM(?: Loss)?[:=] ?(?P<lm>[0-9.]+).*?'
    r'DiscLM(?: Loss)?[:=] ?(?P<disclm>[0-9.]+).*?'
    r'GenLM(?: Loss)?[:=] ?(?P<genlm>[0-9.]+)'
)
MPS_RE = re.compile(r'MPS:? cur=(?P<cur>[0-9.]+)G? drv=(?P<drv>[0-9.]+)G?')
VAL_RE = re.compile(
    r'(?P<time>\d\d:\d\d:\d\d).*?Validation loss[:=] ?(?P<val>[0-9.]+), Dur loss[:=] ?(?P<dur>[0-9.]+), F0 loss[:=] ?(?P<f0>[0-9.]+)'
)


def run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return ''


def parse_log(path: Path):
    if not path.exists():
        return [], []
    lines = path.read_text(errors='ignore').splitlines()
    start = 0
    for i, line in enumerate(lines):
        if 'WavLM loss enabled; SLM adversarial updates disabled' in line or 'SLM adversarial enabled' in line:
            start = i
    rows = []
    vals = []
    pending = None
    for line in lines[start:]:
        m = STEP_RE.search(line)
        if m:
            d = m.groupdict()
            pending = {
                'time': d['time'],
                'epoch': int(d['epoch']),
                'step': int(d['step']),
                'lm': float(d['lm']),
                'disclm': float(d['disclm']),
                'genlm': float(d['genlm']),
            }
            mm = MPS_RE.search(line)
            if mm:
                pending['cur'] = float(mm.group('cur'))
                pending['drv'] = float(mm.group('drv'))
                rows.append(pending)
                pending = None
            continue
        mm = MPS_RE.search(line)
        if mm and pending is not None:
            pending['cur'] = float(mm.group('cur'))
            pending['drv'] = float(mm.group('drv'))
            rows.append(pending)
            pending = None
            continue
        m = VAL_RE.search(line)
        if m:
            d = m.groupdict()
            vals.append((d['time'], float(d['val']), float(d['dur']), float(d['f0'])))
    return rows, vals


def print_log_report(rows, vals, tail: int):
    print('=== training log ===')
    if not rows:
        print('no step rows parsed')
        return
    for r in rows[-tail:]:
        print(
            f"{r['time']} e{r['epoch']} step{r['step']:3d} "
            f"LM={r['lm']:.2f} DLM={r['disclm']:.2f} GLM={r['genlm']:.2f} "
            f"cur={r['cur']:.2f} drv={r['drv']:.2f}"
        )
    first, last = rows[0], rows[-1]
    print(
        f"since restart: cur {first['cur']:.2f}->{last['cur']:.2f} ({last['cur'] - first['cur']:+.2f}G), "
        f"drv {first['drv']:.2f}->{last['drv']:.2f} ({last['drv'] - first['drv']:+.2f}G), points={len(rows)}"
    )
    by_epoch = defaultdict(list)
    for r in rows:
        by_epoch[r['epoch']].append(r)
    print('per epoch:')
    for epoch in sorted(by_epoch):
        a, b = by_epoch[epoch][0], by_epoch[epoch][-1]
        print(
            f"  e{epoch}: step {a['step']}->{b['step']} "
            f"drv {a['drv']:.2f}->{b['drv']:.2f} ({b['drv'] - a['drv']:+.2f}G) "
            f"cur {a['cur']:.2f}->{b['cur']:.2f}"
        )
    if vals:
        print('validations:')
        for t, val, dur, f0 in vals[-8:]:
            print(f'  {t} val={val:.6f} dur={dur:.6f} f0={f0:.6f}')
        best = min(vals, key=lambda x: x[1])
        print(f'best val: {best[1]:.6f} at {best[0]}')


def find_train_pid(pattern: str) -> str | None:
    ps = run(['ps', '-axo', 'pid,ppid,stat,%cpu,%mem,rss,vsz,etime,command'])
    matches = []
    for line in ps.splitlines():
        if pattern in line and 'accelerate launch' not in line and 'report_training_status.py' not in line:
            matches.append(line)
    if matches:
        print('=== train process ===')
        for line in matches:
            print(line)
        return matches[0].split()[0]
    print('=== train process ===')
    print('no matching train process found')
    return None


def print_system_memory():
    print('=== system memory ===')
    swap = run(['sysctl', 'vm.swapusage']).strip()
    if swap:
        print(swap)
    pressure = run(['memory_pressure'])
    for line in pressure.splitlines():
        if 'System-wide memory free percentage' in line:
            print(line.strip())
            break
    vm = run(['vm_stat'])
    page_size = 16384
    vals = {}
    for line in vm.splitlines():
        key = line.split(':')[0].strip()
        m = re.search(r'(\d+)\.?$', line.strip())
        if m:
            vals[key] = int(m.group(1)) * page_size / 1024 ** 3
    keys = [
        ('Pages free', 'free'),
        ('Pages active', 'active'),
        ('Pages inactive', 'inactive'),
        ('Pages wired down', 'wired'),
        ('Pages occupied by compressor', 'comp_occ'),
        ('Pages stored in compressor', 'comp_stored'),
        ('File-backed pages', 'file'),
        ('Anonymous pages', 'anon'),
    ]
    parts = [f'{label} {vals[k]:.2f}G' for k, label in keys if k in vals]
    if parts:
        print(' '.join(parts))


def print_top_processes(limit: int = 8):
    print('=== top memory processes ===')
    ps = run(['ps', '-axo', 'pid,%cpu,%mem,rss,etime,command'])
    rows = []
    for line in ps.splitlines()[1:]:
        parts = line.split(None, 5)
        if len(parts) == 6:
            try:
                rows.append((int(parts[3]), line))
            except ValueError:
                pass
    for _, line in sorted(rows, reverse=True)[:limit]:
        print(line)


def print_vmmap(pid: str):
    print('=== vmmap summary ===')
    out = run(['vmmap', '-summary', pid])
    if not out:
        print('vmmap unavailable')
        return
    interesting = ('Physical footprint', 'MALLOC_SMALL', 'owned unmapped', 'IOAccelerator', 'TOTAL ')
    for line in out.splitlines():
        if any(s in line for s in interesting):
            print(line)


def print_footprint(pid: str):
    print('=== footprint summary ===')
    out = run(['footprint', '-p', pid])
    if not out:
        print('footprint unavailable')
        return
    interesting = ('Footprint', 'Physical', 'Dirty', 'Swapped', 'Compressed', 'MALLOC', 'IOAccelerator', 'VM:')
    shown = 0
    for line in out.splitlines():
        if any(s in line for s in interesting):
            print(line)
            shown += 1
            if shown >= 40:
                break


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', default='runs/vctk_spksep')
    p.add_argument('--log', default=None)
    p.add_argument('--pattern', default='train_second.py --config_path configs/config_vctk_ft.yml')
    p.add_argument('--tail', type=int, default=20)
    p.add_argument('--top', type=int, default=8, help='number of top RSS processes to show')
    p.add_argument('--no-vmmap', action='store_true')
    p.add_argument('--no-footprint', action='store_true')
    args = p.parse_args()

    log_path = Path(args.log) if args.log else Path(args.run_dir) / 'train.log'
    print(f'log: {log_path}')
    rows, vals = parse_log(log_path)
    print_log_report(rows, vals, args.tail)
    print()
    pid = find_train_pid(args.pattern)
    print()
    print_system_memory()
    print()
    if args.top:
        print_top_processes(args.top)
        print()
    if pid and not args.no_vmmap:
        print_vmmap(pid)
        print()
    if pid and not args.no_footprint:
        print_footprint(pid)


if __name__ == '__main__':
    main()
