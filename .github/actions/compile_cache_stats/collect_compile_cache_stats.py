#!/usr/bin/env python3
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Collect compile-cache statistics from sccache or ccache and emit human- + machine-readable output."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from typing import Any


JSON_BEGIN_MARKER = "<!-- compile-cache-stats:json -->"
JSON_END_MARKER = "<!-- /compile-cache-stats:json -->"


def normalize_metric_key(label: str) -> str:
    key = label.strip().lower()
    key = key.replace("(c/c++)", "_c_cpp")
    key = re.sub(r"[()]", "", key)
    key = re.sub(r"[^\w]+", "_", key)
    return key.strip("_")


def _ratio(numerator: float | int | None, denominator: float | int | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _pct(rate: float | None) -> float | None:
    if rate is None:
        return None
    return round(rate * 100.0, 4)


def resolve_sccache() -> str | None:
    path = os.environ.get("SCCACHE_PATH", "").strip()
    if path and os.path.isfile(path) and os.access(path, os.X_OK):
        return path
    return shutil.which("sccache")


def resolve_ccache() -> str | None:
    return shutil.which("ccache")


def run_show_stats(executable: str) -> str:
    result = subprocess.run(
        [executable, "--show-stats"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(
            f"{executable} --show-stats failed with exit code {result.returncode}"
            + (f": {stderr}" if stderr else "")
        )
    return result.stdout


def _parse_sccache_scalar(value: str) -> Any:
    value = value.strip()
    seconds_match = re.fullmatch(r"(\d+(?:\.\d+)?)\s+s", value)
    if seconds_match:
        return float(seconds_match.group(1))
    if re.fullmatch(r"\d+", value):
        return int(value)
    if re.fullmatch(r"\d+(?:\.\d+)?", value):
        return float(value)
    return value


def parse_sccache_stats(stdout: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    non_cacheable_reasons: dict[str, int] = {}

    in_reasons = False
    metadata_labels = {"cache location", "version (client)"}

    for raw_line in stdout.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            in_reasons = False
            continue
        if stripped == "Non-cacheable reasons:":
            in_reasons = True
            continue
        if in_reasons:
            match = re.match(r"^(.+?)\s{2,}(\d+)$", line)
            if match:
                non_cacheable_reasons[normalize_metric_key(match.group(1))] = int(match.group(2))
            continue

        match = re.match(r"^(.+?)\s{2,}(.+)$", line)
        if not match:
            continue
        label, value_raw = match.group(1).strip(), match.group(2).strip()
        label_lower = label.lower()
        parsed = _parse_sccache_scalar(value_raw)
        if label_lower in metadata_labels:
            metadata[normalize_metric_key(label)] = parsed
        else:
            metrics[normalize_metric_key(label)] = parsed

    computed: dict[str, Any] = {}

    hits = metrics.get("cache_hits")
    misses = metrics.get("cache_misses")
    computed["cache_hit_rate"] = _ratio(hits, (hits or 0) + (misses or 0) if hits is not None or misses is not None else None)
    computed["cache_miss_rate"] = _ratio(misses, (hits or 0) + (misses or 0) if hits is not None or misses is not None else None)
    computed["cache_hit_percentage"] = _pct(computed["cache_hit_rate"])
    computed["cache_miss_percentage"] = _pct(computed["cache_miss_rate"])

    hits_cc = metrics.get("cache_hits_c_cpp")
    misses_cc = metrics.get("cache_misses_c_cpp")
    denom_cc = (hits_cc or 0) + (misses_cc or 0) if hits_cc is not None or misses_cc is not None else None
    computed["cache_hit_rate_c_cpp"] = _ratio(hits_cc, denom_cc)
    computed["cache_miss_rate_c_cpp"] = _ratio(misses_cc, denom_cc)
    computed["cache_hit_percentage_c_cpp"] = _pct(computed["cache_hit_rate_c_cpp"])
    computed["cache_miss_percentage_c_cpp"] = _pct(computed["cache_miss_rate_c_cpp"])

    compile_requests = metrics.get("compile_requests")
    compile_executed = metrics.get("compile_requests_executed")
    computed["compile_requests_executed_rate"] = _ratio(compile_executed, compile_requests)
    computed["compile_requests_executed_percentage"] = _pct(computed["compile_requests_executed_rate"])

    cache_errors = metrics.get("cache_errors")
    lookups = (hits or 0) + (misses or 0) if hits is not None or misses is not None else None
    computed["cache_error_rate"] = _ratio(cache_errors, lookups)
    computed["cache_error_percentage"] = _pct(computed["cache_error_rate"])

    cache_errors_cc = metrics.get("cache_errors_c_cpp")
    computed["cache_error_rate_c_cpp"] = _ratio(cache_errors_cc, denom_cc)
    computed["cache_error_percentage_c_cpp"] = _pct(computed["cache_error_rate_c_cpp"])

    compilation_failures = metrics.get("compilation_failures")
    computed["compilation_failure_rate"] = _ratio(compilation_failures, compile_executed)
    computed["compilation_failure_percentage"] = _pct(computed["compilation_failure_rate"])

    non_cacheable_calls = metrics.get("non_cacheable_calls")
    computed["non_cacheable_call_rate"] = _ratio(non_cacheable_calls, compile_requests)
    computed["non_cacheable_call_percentage"] = _pct(computed["non_cacheable_call_rate"])

    non_cacheable_compilations = metrics.get("non_cacheable_compilations")
    computed["non_cacheable_compilation_rate"] = _ratio(non_cacheable_compilations, compile_requests)
    computed["non_cacheable_compilation_percentage"] = _pct(computed["non_cacheable_compilation_rate"])

    return {
        "tool": "sccache",
        "metrics": metrics,
        "computed": {k: v for k, v in computed.items() if v is not None},
        "non_cacheable_reasons": non_cacheable_reasons,
        "metadata": metadata,
        "raw_stdout": stdout,
    }


_RATIO_LINE = re.compile(
    r"^(?P<indent>\s*)(?P<name>[^:]+):\s+"
    r"(?P<num>[\d.]+)\s*/\s*(?P<den>[\d.]+)\s*\((?P<pct>[\d.]+)%\)\s*$"
)
_SINGLE_LINE = re.compile(r"^(?P<indent>\s*)(?P<name>[^:]+):\s+(?P<val>[\d.]+)\s*$")


def _ccache_entry(name: str, numerator: float, denominator: float, percentage: float | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "numerator": numerator,
        "denominator": denominator,
    }
    if percentage is not None:
        entry["percentage"] = percentage
    rate = _ratio(numerator, denominator)
    if rate is not None:
        entry["rate"] = round(rate, 6)
        entry["percentage_computed"] = _pct(rate)
    return entry


def parse_ccache_stats(stdout: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    local_storage: dict[str, Any] = {}
    in_local_storage = False

    for raw_line in stdout.splitlines():
        line = raw_line.rstrip("\n")
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == "Local storage:":
            in_local_storage = True
            continue

        ratio_match = _RATIO_LINE.match(line)
        if ratio_match:
            name = normalize_metric_key(ratio_match.group("name"))
            entry = _ccache_entry(
                ratio_match.group("name"),
                float(ratio_match.group("num")),
                float(ratio_match.group("den")),
                float(ratio_match.group("pct")),
            )
            if in_local_storage:
                local_storage[name] = entry
            else:
                metrics[name] = entry
            continue

        single_match = _SINGLE_LINE.match(line)
        if single_match:
            name = normalize_metric_key(single_match.group("name"))
            value = float(single_match.group("val"))
            if in_local_storage:
                local_storage[name] = value
            else:
                metrics[name] = value

    computed: dict[str, Any] = {}

    cacheable = metrics.get("cacheable_calls")
    if isinstance(cacheable, dict):
        hits = metrics.get("hits")
        misses = metrics.get("misses")
        if isinstance(hits, dict) and isinstance(misses, dict):
            hit_rate = _ratio(hits["numerator"], cacheable["numerator"])
            miss_rate = _ratio(misses["numerator"], cacheable["numerator"])
            computed["cache_hit_rate_of_cacheable"] = hit_rate
            computed["cache_miss_rate_of_cacheable"] = miss_rate
            computed["cache_hit_percentage_of_cacheable"] = _pct(hit_rate)
            computed["cache_miss_percentage_of_cacheable"] = _pct(miss_rate)

            direct = metrics.get("direct")
            preprocessed = metrics.get("preprocessed")
            if isinstance(direct, dict):
                computed["direct_hit_rate_of_hits"] = _ratio(direct["numerator"], hits["numerator"])
                computed["direct_hit_percentage_of_hits"] = _pct(computed["direct_hit_rate_of_hits"])
            if isinstance(preprocessed, dict):
                computed["preprocessed_hit_rate_of_hits"] = _ratio(preprocessed["numerator"], hits["numerator"])
                computed["preprocessed_hit_percentage_of_hits"] = _pct(computed["preprocessed_hit_rate_of_hits"])

        uncacheable = metrics.get("uncacheable_calls")
        if isinstance(uncacheable, dict):
            total_calls = cacheable["denominator"]
            computed["uncacheable_call_rate_of_total"] = _ratio(uncacheable["numerator"], total_calls)
            computed["uncacheable_call_percentage_of_total"] = _pct(computed["uncacheable_call_rate_of_total"])

    cache_size = local_storage.get("cache_size_gb")
    if isinstance(cache_size, dict):
        computed["local_cache_size_utilization_rate"] = _ratio(cache_size["numerator"], cache_size["denominator"])
        computed["local_cache_size_utilization_percentage"] = _pct(computed["local_cache_size_utilization_rate"])

    if isinstance(local_storage.get("hits"), dict) and isinstance(local_storage.get("misses"), dict):
        ls_hits = local_storage["hits"]["numerator"]
        ls_misses = local_storage["misses"]["numerator"]
        computed["local_storage_hit_rate"] = _ratio(ls_hits, ls_hits + ls_misses)
        computed["local_storage_hit_percentage"] = _pct(computed["local_storage_hit_rate"])

    return {
        "tool": "ccache",
        "metrics": metrics,
        "local_storage": local_storage,
        "computed": {k: v for k, v in computed.items() if v is not None},
        "raw_stdout": stdout,
    }


def format_human_readable(report: dict[str, Any]) -> str:
    lines = [
        "======== Compile cache statistics ========",
        f"Tool: {report['tool']}",
        "",
    ]

    if report["tool"] == "sccache":
        lines.append("--- Metrics ---")
        for key in sorted(report["metrics"]):
            lines.append(f"  {key}: {report['metrics'][key]}")
        if report.get("non_cacheable_reasons"):
            lines.append("")
            lines.append("--- Non-cacheable reasons ---")
            for key, value in sorted(report["non_cacheable_reasons"].items()):
                lines.append(f"  {key}: {value}")
        if report.get("metadata"):
            lines.append("")
            lines.append("--- Metadata ---")
            for key, value in sorted(report["metadata"].items()):
                lines.append(f"  {key}: {value}")
    else:
        lines.append("--- Metrics ---")
        for key, value in sorted(report["metrics"].items()):
            lines.append(f"  {key}: {_format_ccache_value(value)}")
        if report.get("local_storage"):
            lines.append("")
            lines.append("--- Local storage ---")
            for key, value in sorted(report["local_storage"].items()):
                lines.append(f"  {key}: {_format_ccache_value(value)}")

    if report.get("computed"):
        lines.append("")
        lines.append("--- Computed ---")
        for key in sorted(report["computed"]):
            value = report["computed"][key]
            if isinstance(value, float) and "percentage" in key:
                lines.append(f"  {key}: {value}%")
            elif isinstance(value, float) and "rate" in key:
                lines.append(f"  {key}: {value:.6f}")
            else:
                lines.append(f"  {key}: {value}")

    lines.append("==========================================")
    return "\n".join(lines)


def _format_ccache_value(value: Any) -> str:
    if isinstance(value, dict):
        parts = [f"{value['numerator']}/{value['denominator']}"]
        if "percentage" in value:
            parts.append(f"({value['percentage']}%)")
        if "percentage_computed" in value and "percentage" not in value:
            parts.append(f"(computed {value['percentage_computed']}%)")
        return " ".join(parts)
    return str(value)


def set_github_output(name: str, value: str) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    with open(output_path, "a", encoding="utf-8") as handle:
        delimiter = f"compile_cache_stats_{name}"
        handle.write(f"{name}<<{delimiter}\n{value}\n{delimiter}\n")


def append_step_summary(text: str) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    with open(summary_path, "a", encoding="utf-8") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")


def _ci_context() -> dict[str, Any]:
    keys = (
        "GITHUB_RUN_ID",
        "GITHUB_RUN_ATTEMPT",
        "GITHUB_WORKFLOW",
        "GITHUB_JOB",
        "GITHUB_REF",
        "GITHUB_SHA",
        "GITHUB_REPOSITORY",
        "RUNNER_OS",
        "RUNNER_ARCH",
    )
    return {key.lower(): os.environ.get(key) for key in keys if os.environ.get(key)}


def collect() -> dict[str, Any]:
    executable: str | None = None
    tool = ""
    sccache = resolve_sccache()
    if sccache:
        executable = sccache
        tool = "sccache"
    elif resolve_ccache():
        ccache = resolve_ccache()
        executable = ccache
        tool = "ccache"

    if not executable:
        raise FileNotFoundError(
            "Neither sccache nor ccache was found (checked SCCACHE_PATH, PATH for sccache, then PATH for ccache)."
        )

    stdout = run_show_stats(executable)
    if tool == "sccache":
        report = parse_sccache_stats(stdout)
    else:
        report = parse_ccache_stats(stdout)
    report["executable"] = executable
    report["ci"] = _ci_context()
    return report


def write_stats_json(report: dict[str, Any], path: str) -> str:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def main() -> int:
    try:
        report = collect()
    except FileNotFoundError as exc:
        print(f"::error::{exc}")
        return 1
    except RuntimeError as exc:
        print(f"::error::{exc}")
        return 1

    human = format_human_readable(report)
    print(human)
    print("")
    print(JSON_BEGIN_MARKER)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(JSON_END_MARKER)

    stats_json = json.dumps(report, sort_keys=True)
    json_path = os.environ.get("COMPILE_CACHE_STATS_JSON_PATH", "").strip()
    if not json_path:
        json_path = os.path.join(os.environ.get("RUNNER_TEMP", "."), "compile-cache-stats.json")
    json_path = write_stats_json(report, json_path)

    set_github_output("tool", report["tool"])
    set_github_output("stats-json", stats_json)
    set_github_output("stats-json-path", json_path)

    append_step_summary("## Compile cache statistics\n\n")
    append_step_summary(f"**Tool:** `{report['tool']}` (`{report['executable']}`)\n\n")
    append_step_summary("```json\n")
    append_step_summary(json.dumps(report, indent=2, sort_keys=True))
    append_step_summary("\n```\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
