#!/usr/bin/env python3
import argparse
import json
import os
from collections import defaultdict

def load_detailed(report_path):
    with open(report_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    perfs = data.get('detailed_performance', [])
    if not perfs:
        return []
    return perfs[0].get('nodes', [])

def summarize(report_root):
    summary = {}
    if not os.path.isdir(report_root):
        return summary
    for model in sorted(os.listdir(report_root)):
        model_dir = os.path.join(report_root, model)
        if not os.path.isdir(model_dir):
            continue
        report = os.path.join(model_dir, 'benchmark_detailed_counters_report.json')
        if not os.path.exists(report):
            continue
        nodes = load_detailed(report)
        convs = [n for n in nodes if n.get('status') == 'EXECUTED' and n.get('node_type') in ('Convolution', 'GroupConvolution', 'Deconvolution')]
        total_real = sum(float(n.get('real_time', 0.0) or 0.0) for n in convs)
        total_cpu = sum(float(n.get('cpu_time', 0.0) or 0.0) for n in convs)
        by_exec = defaultdict(float)
        for n in convs:
            by_exec[n.get('exec_type', 'unknown')] += float(n.get('real_time', 0.0) or 0.0)
        top = sorted(convs, key=lambda n: float(n.get('real_time', 0.0) or 0.0), reverse=True)[:10]
        summary[model] = {
            'conv_count': len(convs),
            'conv_real_total_ms': total_real,
            'conv_cpu_total_ms': total_cpu,
            'by_exec_type_ms': dict(sorted(by_exec.items(), key=lambda kv: kv[1], reverse=True)),
            'top_convs': [
                {
                    'name': n.get('name'),
                    'node_type': n.get('node_type'),
                    'exec_type': n.get('exec_type'),
                    'real_time_ms': float(n.get('real_time', 0.0) or 0.0),
                    'cpu_time_ms': float(n.get('cpu_time', 0.0) or 0.0),
                } for n in top
            ],
        }
    return summary

def write_text(summary, out_path):
    lines = []
    for model, info in summary.items():
        lines.append(f"{model}:")
        lines.append(f"  conv_count: {info['conv_count']}")
        lines.append(f"  conv_real_total_ms: {info['conv_real_total_ms']:.3f}")
        lines.append(f"  conv_cpu_total_ms: {info['conv_cpu_total_ms']:.3f}")
        lines.append("  by_exec_type_ms:")
        for k, v in info['by_exec_type_ms'].items():
            lines.append(f"    {k}: {v:.3f}")
        lines.append("  top_convs:")
        for n in info['top_convs']:
            lines.append(f"    {n['name']}: {n['exec_type']} {n['real_time_ms']:.3f} ms")
        lines.append("")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

def compare(base, other):
    cmp = {}
    all_models = sorted(set(base) | set(other))
    for m in all_models:
        b = base.get(m)
        o = other.get(m)
        if not b or not o:
            cmp[m] = {'status': 'missing', 'base_present': bool(b), 'other_present': bool(o)}
            continue
        diff_total = b['conv_real_total_ms'] - o['conv_real_total_ms']
        diff_exec = {}
        keys = set(b['by_exec_type_ms']) | set(o['by_exec_type_ms'])
        for k in keys:
            diff_exec[k] = b['by_exec_type_ms'].get(k, 0.0) - o['by_exec_type_ms'].get(k, 0.0)
        cmp[m] = {
            'status': 'ok',
            'conv_real_total_ms_delta': diff_total,
            'conv_cpu_total_ms_delta': b['conv_cpu_total_ms'] - o['conv_cpu_total_ms'],
            'by_exec_type_ms_delta': dict(sorted(diff_exec.items(), key=lambda kv: kv[1], reverse=True)),
        }
    return cmp

def write_compare_text(cmp, out_path):
    lines = []
    for model, info in cmp.items():
        lines.append(f"{model}:")
        if info['status'] != 'ok':
            lines.append(f"  status: {info['status']} base_present={info.get('base_present')} other_present={info.get('other_present')}")
            lines.append("")
            continue
        lines.append(f"  conv_real_total_ms_delta: {info['conv_real_total_ms_delta']:.3f}")
        lines.append(f"  conv_cpu_total_ms_delta: {info['conv_cpu_total_ms_delta']:.3f}")
        lines.append("  by_exec_type_ms_delta:")
        for k, v in info['by_exec_type_ms_delta'].items():
            lines.append(f"    {k}: {v:.3f}")
        lines.append("")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--report-root', required=True)
    ap.add_argument('--out-prefix', required=True)
    ap.add_argument('--compare', default=None, help='Path to other summary json to compare')
    args = ap.parse_args()

    summary = summarize(args.report_root)
    json_path = f"{args.out_prefix}_conv_summary.json"
    txt_path = f"{args.out_prefix}_conv_summary.txt"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    write_text(summary, txt_path)

    if args.compare:
        with open(args.compare, 'r', encoding='utf-8') as f:
            other = json.load(f)
        cmp = compare(summary, other)
        cmp_json = f"{args.out_prefix}_conv_compare.json"
        cmp_txt = f"{args.out_prefix}_conv_compare.txt"
        with open(cmp_json, 'w', encoding='utf-8') as f:
            json.dump(cmp, f, indent=2)
        write_compare_text(cmp, cmp_txt)

if __name__ == '__main__':
    main()
