#!/usr/bin/env python3
import csv
import sys
from collections import defaultdict

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'gemmv_vs_benchdnn_ext.csv'
    rows = []
    with open(path, newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            # suite: benchdnn/ours; impl: bench impl or ours; M,K,N,time_ms,gflops
            rows.append(row)
    # Build maps: benchdnn impl -> {(M,K): (time_ms,gflops)}; ours -> {(M,K): (time_ms,gflops)}
    bench = defaultdict(dict)
    ours = {}
    for row in rows:
        M = int(row['M']); K = int(row['K'])
        key = (M, K)
        if row['suite'] == 'benchdnn':
            impl = row['impl']
            bench[impl][key] = (float(row['time_ms']), float(row['gflops']))
        else:
            ours[key] = (float(row['time_ms']), float(row['gflops']))
    # Map fuzzy categories: VNNI and AMX. Bench implementations vary between versions.
    categories = {
        'VNNI': lambda impl: 'vnni' in impl.lower(),
        'AMX':  lambda impl: 'amx' in impl.lower(),
    }
    for cat, pred in categories.items():
        # Build merged dict of all impls matching predicate
        catbench = {}
        for impl, mp in bench.items():
            if pred(impl):
                catbench.update(mp)
        num=0; s_ratio=0.0
        per_shape = []
        for key, (t_ours, g_ours) in ours.items():
            if key in catbench:
                t_b, g_b = catbench[key]
                if g_b > 0:
                    r = g_ours / g_b
                    s_ratio += r; num += 1
                    per_shape.append((key[0], key[1], r))
        avg = (s_ratio/num) if num else 0.0
        print(f'{cat}: avg_ratio={avg:.3f} over {len(per_shape)} shapes')
        for M,K,r in sorted(per_shape):
            print(f'  M={M:4d} K={K:4d} ours/bench={r:.3f}')

if __name__ == '__main__':
    main()
