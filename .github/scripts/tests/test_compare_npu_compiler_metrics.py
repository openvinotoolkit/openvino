# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess  # nosec B404
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "compare_npu_compiler_metrics.py"


def metrics(memory, time):
    return {"compilation_memory_usage_kb": memory, "compile_net_time_ms": time}


def wrap(models, platform="3720", framework="pytorch", test_type="pt_npu_models"):
    return {platform: {framework: {test_type: models}}}


def run_compare(tmp_path: Path, baseline, current, *, write_baseline=True, extra_args=()):
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    output_path = tmp_path / "diff.json"
    summary_path = tmp_path / "summary.md"
    if write_baseline:
        baseline_path.write_text(baseline if isinstance(baseline, str) else json.dumps(baseline), encoding="utf-8")
    current_path.write_text(current if isinstance(current, str) else json.dumps(current), encoding="utf-8")
    result = subprocess.run(  # nosec B603
        [
            sys.executable,
            str(SCRIPT),
            "--baseline",
            str(baseline_path),
            "--current",
            str(current_path),
            "--output",
            str(output_path),
            *extra_args,
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={**os.environ, "GITHUB_STEP_SUMMARY": str(summary_path)},
    )
    return result, output_path, summary_path


def statuses(output_path: Path):
    return {entry["model"]: entry["status"] for entry in json.loads(output_path.read_text(encoding="utf-8"))}


def test_missing_baseline_is_not_an_error(tmp_path):
    result, output_path, summary_path = run_compare(tmp_path, None, wrap({"a": metrics(100, 10)}), write_baseline=False)

    assert result.returncode == 0, result.stderr
    assert not output_path.exists()
    assert "No baseline" in summary_path.read_text(encoding="utf-8")


def test_identical_files_are_unchanged(tmp_path):
    data = wrap({"a": metrics(100, 10)})
    result, output_path, _ = run_compare(tmp_path, data, data)

    assert result.returncode == 0, result.stderr
    assert statuses(output_path) == {"a": "unchanged"}


def test_regression_and_improvement_beyond_threshold(tmp_path):
    baseline = wrap({"slower": metrics(100, 100), "smaller": metrics(100, 100)})
    current = wrap({"slower": metrics(120, 100), "smaller": metrics(80, 100)})
    result, output_path, summary_path = run_compare(tmp_path, baseline, current)

    assert result.returncode == 0, result.stderr
    assert statuses(output_path) == {"slower": "regressed", "smaller": "improved"}
    assert "slower" in summary_path.read_text(encoding="utf-8")


def test_change_below_noise_floor_is_unchanged(tmp_path):
    # 4% memory (floor 5%) and 10% time (floor 15%).
    result, output_path, _ = run_compare(tmp_path, wrap({"a": metrics(100, 100)}), wrap({"a": metrics(104, 110)}))

    assert result.returncode == 0, result.stderr
    assert statuses(output_path) == {"a": "unchanged"}


def test_thresholds_are_configurable(tmp_path):
    result, output_path, _ = run_compare(
        tmp_path,
        wrap({"a": metrics(100, 100)}),
        wrap({"a": metrics(104, 100)}),
        extra_args=("--memory-threshold", "1"),
    )

    assert result.returncode == 0, result.stderr
    assert statuses(output_path) == {"a": "regressed"}


def test_new_and_removed_models(tmp_path):
    result, output_path, _ = run_compare(tmp_path, wrap({"old": metrics(1, 1)}), wrap({"fresh": metrics(1, 1)}))

    assert result.returncode == 0, result.stderr
    assert statuses(output_path) == {"old": "removed", "fresh": "new"}


def test_null_metrics_are_not_compared(tmp_path):
    result, output_path, _ = run_compare(tmp_path, wrap({"a": metrics(None, 100)}), wrap({"a": metrics(500, 100)}))

    assert result.returncode == 0, result.stderr
    diff = json.loads(output_path.read_text(encoding="utf-8"))
    assert diff[0]["metrics"]["compilation_memory_usage_kb"]["status"] == "no_data"
    assert diff[0]["status"] == "unchanged"


def test_invalid_baseline_json_fails(tmp_path):
    result, _, _ = run_compare(tmp_path, "{not json", wrap({"a": metrics(1, 1)}))

    assert result.returncode != 0
    assert "Cannot read" in result.stderr
