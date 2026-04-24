#!/usr/bin/env python3
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def load_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            if row["execStatus"] != "EXECUTED":
                continue
            rows.append(
                {
                    "layer": row["layerName"],
                    "layer_type": row["layerType"],
                    "exec_type": row["execType"],
                    "real": float(row["realTime (ms)"]),
                }
            )
    return rows


def summarize_exec(rows: list[dict[str, object]]) -> list[tuple[str, float]]:
    totals: dict[str, float] = defaultdict(float)
    for row in rows:
        totals[str(row["exec_type"])] += float(row["real"])
    return sorted(totals.items(), key=lambda kv: kv[1], reverse=True)


def top_rows(rows: list[dict[str, object]], exec_type: str, limit: int) -> list[tuple[str, float]]:
    filtered = [row for row in rows if row["exec_type"] == exec_type]
    filtered.sort(key=lambda row: float(row["real"]), reverse=True)
    return [(str(row["layer"]), float(row["real"])) for row in filtered[:limit]]


def diff_rows(
    base: list[dict[str, object]],
    new: list[dict[str, object]],
    min_abs_diff: float,
) -> tuple[list[tuple[float, float, float, str, str]], list[tuple[float, float, float, str, str]]]:
    base_map = {(str(r["layer"]), str(r["exec_type"])): float(r["real"]) for r in base}
    new_map = {(str(r["layer"]), str(r["exec_type"])): float(r["real"]) for r in new}
    deltas: list[tuple[float, float, float, str, str]] = []
    for key in sorted(set(base_map) | set(new_map)):
        before = base_map.get(key, 0.0)
        after = new_map.get(key, 0.0)
        delta = after - before
        if abs(delta) < min_abs_diff:
            continue
        deltas.append((delta, before, after, key[1], key[0]))
    regressions = sorted((d for d in deltas if d[0] > 0.0), reverse=True)
    improvements = sorted((d for d in deltas if d[0] < 0.0))
    return regressions, improvements


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two benchmark_app detailed counters CSV reports.")
    parser.add_argument("base", type=Path, help="Baseline benchmark_detailed_counters_report.csv")
    parser.add_argument("new", type=Path, help="Current benchmark_detailed_counters_report.csv")
    parser.add_argument("--limit", type=int, default=12, help="Number of rows to show in each section")
    parser.add_argument("--min-abs-diff", type=float, default=0.03, help="Minimum delta in ms to report")
    args = parser.parse_args()

    base_rows = load_rows(args.base)
    new_rows = load_rows(args.new)

    print("Base:", args.base)
    print("New :", args.new)
    print()

    print("ExecType totals (new)")
    for name, total in summarize_exec(new_rows)[: args.limit]:
        print(f"  {name}: {total:.3f} ms")
    print()

    print("Top brgconv_uni_I8 (new)")
    for name, total in top_rows(new_rows, "brgconv_uni_I8", args.limit):
        print(f"  {name}: {total:.3f} ms")
    print()

    print("Top acl_I8 converts (new)")
    for name, total in top_rows(new_rows, "acl_I8", args.limit):
        print(f"  {name}: {total:.3f} ms")
    print()

    regressions, improvements = diff_rows(base_rows, new_rows, args.min_abs_diff)

    print("Top regressions")
    for delta, before, after, exec_type, layer in regressions[: args.limit]:
        print(f"  {delta:+.3f} ms | {before:.3f} -> {after:.3f} | {exec_type} | {layer}")
    print()

    print("Top improvements")
    for delta, before, after, exec_type, layer in improvements[: args.limit]:
        print(f"  {delta:+.3f} ms | {before:.3f} -> {after:.3f} | {exec_type} | {layer}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
