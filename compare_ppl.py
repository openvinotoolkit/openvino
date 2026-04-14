# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Compare perplexity across KV cache configurations.
#
# Usage:
#   # Compare two configs
#   python compare_ppl.py --model ~/models/Qwen3-8B/ \
#       --baseline ov-configs/config_k_u8_v_u8.json \
#       --target ov-configs/config_k_tbq4_v_tbq4.json
#
#   # Compare all configs in a folder against a baseline
#   python compare_ppl.py --model ~/models/Qwen3-8B/ \
#       --baseline ov-configs/config_k_u8_v_u8.json \
#       --all-from ov-configs/
#
#   # Save results to CSV
#   python compare_ppl.py --model ~/models/Qwen3-8B/ \
#       --all-from ov-configs/ --csv results.csv

import argparse
import csv
import json
import os
import sys

from measure_ppl import measure_perplexity, tokenize_wikitext

import openvino_genai as ov_genai


def load_ov_properties(config_path):
    """Load OV properties from a JSON config file."""
    with open(config_path) as f:
        raw = json.load(f)
    genai_only = {"ATTENTION_BACKEND", "SCHEDULER_CONFIG"}
    return {k: str(v) for k, v in raw.items() if k not in genai_only}


def config_name(path):
    """Extract a short name from config path: config_k_u8_v_u8.json -> k_u8_v_u8."""
    name = os.path.basename(path)
    if name.startswith("config_"):
        name = name[len("config_"):]
    if name.endswith(".json"):
        name = name[:-len(".json")]
    return name


def parse_kv_from_config(config_path):
    """Extract K and V mode strings from config."""
    with open(config_path) as f:
        raw = json.load(f)
    k_mode = raw.get("KEY_CACHE_CODEC", raw.get("KEY_CACHE_PRECISION", "?"))
    v_mode = raw.get("VALUE_CACHE_CODEC", raw.get("VALUE_CACHE_PRECISION", "?"))
    return str(k_mode), str(v_mode)


def run_one(model_path, tokens, window, prefill, config_path,
            inference_precision=None):
    """Run PPL measurement for one config, return result dict + metadata."""
    props = load_ov_properties(config_path) if config_path else {}
    if inference_precision:
        props["INFERENCE_PRECISION_HINT"] = inference_precision
    name = config_name(config_path) if config_path else "no_config"
    k_mode, v_mode = parse_kv_from_config(config_path) if config_path else ("f32", "f32")

    print(f"\n{'='*60}")
    print(f"  {name}  (K={k_mode}, V={v_mode})")
    print(f"{'='*60}")

    result = measure_perplexity(model_path, tokens, window, prefill, props)
    result["name"] = name
    result["k_mode"] = k_mode
    result["v_mode"] = v_mode
    result["config"] = config_path or ""
    return result


def print_table(results, baseline_decode_ppl):
    """Print a formatted comparison table."""
    print(f"\n{'='*100}")
    print(f"{'Config':<30} {'K':<10} {'V':<10} {'Decode PPL':>12} {'vs baseline':>12} {'Time':>8}  {'Status'}")
    print(f"{'-'*100}")
    for r in results:
        if r.get("error"):
            print(f"{r['name']:<30} {r['k_mode']:<10} {r['v_mode']:<10} "
                  f"{'FAILED':>12} {'—':>12} {'—':>8}  {r['error'][:40]}")
        else:
            delta = (r["decode_ppl"] / baseline_decode_ppl - 1) * 100 if baseline_decode_ppl > 0 else 0
            sign = "+" if delta >= 0 else ""
            print(f"{r['name']:<30} {r['k_mode']:<10} {r['v_mode']:<10} "
                  f"{r['decode_ppl']:>12.4f} {sign}{delta:>10.1f}% {r['elapsed']:>7.1f}s")
    print(f"{'='*100}")


def write_csv(results, baseline_decode_ppl, csv_path):
    """Write results to CSV."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "k_mode", "v_mode", "decode_ppl", "prefill_ppl",
                         "ppl", "vs_baseline_pct", "decode_tokens", "prefill_tokens",
                         "elapsed_s", "error"])
        for r in results:
            error = r.get("error", "")
            if error:
                writer.writerow([
                    r["name"], r["k_mode"], r["v_mode"],
                    "FAILED", "FAILED", "FAILED",
                    "—", 0, 0, "0.0", error,
                ])
            else:
                delta = (r["decode_ppl"] / baseline_decode_ppl - 1) * 100 if baseline_decode_ppl > 0 else 0
                writer.writerow([
                    r["name"], r["k_mode"], r["v_mode"],
                    f"{r['decode_ppl']:.4f}", f"{r['prefill_ppl']:.4f}", f"{r['ppl']:.4f}",
                    f"{delta:+.2f}%", r["decode_tokens"], r["prefill_tokens"],
                    f"{r['elapsed']:.1f}", "",
                ])
    print(f"\nCSV written to {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare perplexity across KV cache configurations")
    parser.add_argument("--model", required=True,
                        help="Path to OpenVINO model directory")
    parser.add_argument("--baseline", default=None,
                        help="Baseline config JSON (default: first config or no-config)")
    parser.add_argument("--target", default=None, nargs="+",
                        help="Target config JSON(s) to compare against baseline")
    parser.add_argument("--all-from", default=None,
                        help="Directory of config JSONs to compare (all *.json files)")
    parser.add_argument("--window", type=int, default=2048,
                        help="Chunk size in tokens (default: 2048)")
    parser.add_argument("--prefill", type=int, default=0,
                        help="Prefill tokens per chunk (0=auto=window-1)")
    parser.add_argument("--max-tokens", type=int, default=0,
                        help="Max tokens from dataset (0=all)")
    parser.add_argument("--csv", default=None,
                        help="Output CSV file path")
    parser.add_argument("--inference-precision", default=None,
                        help="Inference precision (e.g. f32, bf16). "
                             "Sets INFERENCE_PRECISION_HINT property.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick check that every config compiles and runs "
                             "(1 prefill + 1 decode token per config)")
    args = parser.parse_args()

    if args.smoke_test:
        args.window = 3
        args.prefill = 1
        args.max_tokens = 3

    if args.prefill == 0:
        args.prefill = args.window - 1

    # Collect config files to run
    configs = []
    if args.baseline:
        configs.append(args.baseline)
    if args.target:
        configs.extend(args.target)
    if args.all_from:
        folder = args.all_from
        jsons = sorted(f for f in os.listdir(folder) if f.endswith(".json"))
        for j in jsons:
            path = os.path.join(folder, j)
            if path not in configs:
                configs.append(path)

    if not configs:
        print("Error: provide --baseline/--target or --all-from", file=sys.stderr)
        return 1

    # Tokenize once
    print(f"Loading tokenizer from {args.model}...")
    tokenizer = ov_genai.Tokenizer(args.model)
    print("Tokenizing WikiText-2 test set...")
    tokens = tokenize_wikitext(tokenizer, args.max_tokens)
    print(f"  {len(tokens)} tokens")

    # Run all configs sequentially
    results = []
    for cfg in configs:
        try:
            r = run_one(args.model, tokens, args.window, args.prefill, cfg,
                        args.inference_precision)
            results.append(r)
        except Exception as e:
            name = config_name(cfg)
            k_mode, v_mode = parse_kv_from_config(cfg) if cfg else ("?", "?")
            print(f"\n  FAILED: {name} — {e}\n")
            results.append({
                "name": name,
                "k_mode": k_mode,
                "v_mode": v_mode,
                "config": cfg or "",
                "ppl": float("inf"),
                "decode_ppl": float("inf"),
                "prefill_ppl": float("inf"),
                "total_tokens": 0,
                "decode_tokens": 0,
                "prefill_tokens": 0,
                "elapsed": 0.0,
                "error": str(e),
            })

    successful = [r for r in results if not r.get("error")]
    failed = [r for r in results if r.get("error")]

    if not successful:
        print("All configs failed, nothing to report.")
        print_table(results, 1.0)
        return 1

    # Determine baseline: explicit --baseline, or best (lowest) decode PPL
    if args.baseline:
        baseline_name = config_name(args.baseline)
        baseline_decode_ppl = next(
            (r["decode_ppl"] for r in successful if r["name"] == baseline_name),
            successful[0]["decode_ppl"])
    else:
        baseline_decode_ppl = min(r["decode_ppl"] for r in successful)

    # Output
    print_table(results, baseline_decode_ppl)
    if args.csv:
        write_csv(results, baseline_decode_ppl, args.csv)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
