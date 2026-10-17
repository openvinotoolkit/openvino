# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Compare per-model NPU compiler metrics of the current run against a baseline.

Both inputs use the layout written by parse_npu_compiler_log.py:

    platform -> framework -> test_type -> model -> {compilation_memory_usage_kb, compile_net_time_ms}

The comparison is report-only: a missing baseline or a regression never produces a
non-zero exit code. Only unreadable or structurally invalid input does.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

METRIC_KEYS = ("compilation_memory_usage_kb", "compile_net_time_ms")
# Relative change (percent) below which a metric is considered noise.
DEFAULT_THRESHOLDS = {"compilation_memory_usage_kb": 5.0, "compile_net_time_ms": 15.0}
TOP_N = 20


def append_step_summary(text: str) -> None:
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(text)


def load_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as err:
        raise SystemExit(f"Cannot read '{path}': {err}")
    if not isinstance(data, dict):
        raise SystemExit(f"'{path}' does not contain a JSON object")
    return data


def flatten(data: dict, path: Path) -> dict[tuple[str, str, str, str], dict]:
    """Turn platform/framework/test_type/model nesting into {(p, f, t, m): metrics}."""
    flat = {}
    for platform, frameworks in data.items():
        for framework, test_types in _as_dict(frameworks, path).items():
            for test_type, models in _as_dict(test_types, path).items():
                for model, metrics in _as_dict(models, path).items():
                    flat[(platform, framework, test_type, model)] = _as_dict(metrics, path)
    return flat


def _as_dict(value: object, path: Path) -> dict:
    if not isinstance(value, dict):
        raise SystemExit(f"'{path}' has an unexpected structure (expected nested JSON objects)")
    return value


def compare_metric(base: object, cur: object, threshold: float) -> dict:
    if not isinstance(base, (int, float)) or not isinstance(cur, (int, float)):
        return {"baseline": base, "current": cur, "status": "no_data"}
    delta = cur - base
    percent = (delta / base * 100.0) if base else (0.0 if delta == 0 else None)
    if percent is None:
        status = "regressed" if delta > 0 else "improved"
    elif percent > threshold:
        status = "regressed"
    elif percent < -threshold:
        status = "improved"
    else:
        status = "unchanged"
    return {
        "baseline": base,
        "current": cur,
        "delta": delta,
        "percent": None if percent is None else round(percent, 2),
        "status": status,
    }


def compare(baseline: dict, current: dict, thresholds: dict[str, float], baseline_path: Path, current_path: Path) -> list[dict]:
    base_flat = flatten(baseline, baseline_path)
    cur_flat = flatten(current, current_path)
    entries = []
    for key in sorted(set(base_flat) | set(cur_flat)):
        platform, framework, test_type, model = key
        entry = {"platform": platform, "framework": framework, "test_type": test_type, "model": model}
        if key not in base_flat:
            entry["status"] = "new"
        elif key not in cur_flat:
            entry["status"] = "removed"
        else:
            metrics = {
                name: compare_metric(base_flat[key].get(name), cur_flat[key].get(name), thresholds[name])
                for name in METRIC_KEYS
            }
            statuses = {m["status"] for m in metrics.values()}
            entry["metrics"] = metrics
            entry["status"] = (
                "regressed" if "regressed" in statuses else "improved" if "improved" in statuses else "unchanged"
            )
        entries.append(entry)
    return entries


def worst_percent(entry: dict) -> float:
    percents = [m.get("percent") for m in entry.get("metrics", {}).values()]
    return max((p for p in percents if p is not None), default=0.0)


def format_cell(metric: dict) -> str:
    if metric["status"] == "no_data":
        return "n/a"
    percent = metric["percent"]
    pct = "n/a" if percent is None else f"{percent:+.1f}%"
    return f"{metric['baseline']:g} -> {metric['current']:g} ({pct})"


def render_summary(entries: list[dict], title: str) -> str:
    counts = {status: 0 for status in ("regressed", "improved", "unchanged", "new", "removed")}
    for entry in entries:
        counts[entry["status"]] += 1
    lines = [
        f"### {title}",
        "",
        " | ".join(f"{status}: {count}" for status, count in counts.items()),
        "",
    ]
    regressed = sorted((e for e in entries if e["status"] == "regressed"), key=worst_percent, reverse=True)
    if regressed:
        lines += [
            f"Top {min(TOP_N, len(regressed))} regressions (of {len(regressed)}):",
            "",
            "| Model | Memory (KB) | Compile time (ms) |",
            "|---|---|---|",
        ]
        for entry in regressed[:TOP_N]:
            name = f"{entry['framework']}/{entry['test_type']}/{entry['model']}"
            memory = format_cell(entry["metrics"]["compilation_memory_usage_kb"])
            time = format_cell(entry["metrics"]["compile_net_time_ms"])
            lines.append(f"| {name} | {memory} | {time} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path, help="Metrics JSON restored from the cache")
    parser.add_argument("--current", required=True, type=Path, help="Metrics JSON of this run")
    parser.add_argument("--output", required=True, type=Path, help="Where to write the full diff JSON")
    parser.add_argument("--title", default="NPU compiler metrics vs. baseline", help="Step summary heading")
    parser.add_argument("--memory-threshold", type=float, default=DEFAULT_THRESHOLDS["compilation_memory_usage_kb"],
                        help="Percent change in memory treated as noise")
    parser.add_argument("--time-threshold", type=float, default=DEFAULT_THRESHOLDS["compile_net_time_ms"],
                        help="Percent change in compile time treated as noise")
    args = parser.parse_args()

    if not args.current.is_file():
        raise SystemExit(f"Current metrics file '{args.current}' does not exist")

    if not args.baseline.is_file():
        message = f"No baseline found at '{args.baseline}'; skipping comparison."
        print(message)
        append_step_summary(f"### {args.title}\n\n{message}\n")
        return

    thresholds = {
        "compilation_memory_usage_kb": args.memory_threshold,
        "compile_net_time_ms": args.time_threshold,
    }
    entries = compare(load_json(args.baseline), load_json(args.current), thresholds, args.baseline, args.current)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")

    summary = render_summary(entries, args.title)
    print(summary)
    append_step_summary(summary)


if __name__ == "__main__":
    main()
