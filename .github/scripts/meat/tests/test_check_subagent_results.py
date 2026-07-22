# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for check_subagent_results.py.

The script scans five hard-coded result JSON files and exits 1 if ANY of them:
  - carries  status == "failed"
  - has  test_results  containing the word "FAILED"
  - contains malformed JSON

Missing files are silently skipped (not present ≠ failed).

Exit 0 + "PASS" message means every present file is healthy.
"""

import json
import pathlib

import pytest

from conftest import SCRIPTS_DIR, run_script, write_json

SCRIPT = SCRIPTS_DIR / "check_subagent_results.py"

RESULT_PATHS = [
    "agent-results/frontend/fe_result.json",
    "agent-results/core-opspec/core_opspec_result.json",
    "agent-results/transformation/transformation_result.json",
    "agent-results/cpu/cpu_result.json",
    "agent-results/gpu/gpu_result.json",
]


def _run(tmp_path):
    return run_script(SCRIPT, tmp_path)


def _write(tmp_path, rel_path, data):
    write_json(tmp_path / rel_path, data)


# ---------------------------------------------------------------------------
# No files present
# ---------------------------------------------------------------------------

def test_no_files_exits_zero(tmp_path):
    result = _run(tmp_path)
    assert result.returncode == 0


def test_no_files_prints_pass(tmp_path):
    result = _run(tmp_path)
    assert "PASS" in result.stdout


# ---------------------------------------------------------------------------
# All files present and healthy → exit 0
# ---------------------------------------------------------------------------

def test_all_passing_exits_zero(tmp_path):
    for p in RESULT_PATHS:
        _write(tmp_path, p, {"status": "success", "test_results": "5 passed"})
    assert _run(tmp_path).returncode == 0


# ---------------------------------------------------------------------------
# status == "failed" in any file → exit 1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", RESULT_PATHS)
def test_failed_status_in_one_file_exits_one(tmp_path, path):
    """Any result file with status=failed is sufficient to exit 1."""
    _write(tmp_path, path, {"status": "failed"})
    assert _run(tmp_path).returncode == 1


def test_failed_status_message_printed_to_stdout(tmp_path):
    _write(tmp_path, RESULT_PATHS[0], {"status": "failed"})
    result = _run(tmp_path)
    assert "FAIL" in result.stdout.upper()


def test_mixed_pass_and_fail_exits_one(tmp_path):
    _write(tmp_path, RESULT_PATHS[0], {"status": "success"})
    _write(tmp_path, RESULT_PATHS[1], {"status": "failed"})
    assert _run(tmp_path).returncode == 1


# ---------------------------------------------------------------------------
# "FAILED" in test_results → exit 1
# ---------------------------------------------------------------------------

def test_failed_word_in_test_results_exits_one(tmp_path):
    _write(tmp_path, RESULT_PATHS[2], {"status": "success", "test_results": "3 FAILED, 7 passed"})
    assert _run(tmp_path).returncode == 1


def test_failed_lowercase_in_test_results_exits_one(tmp_path):
    """Case-insensitive: 'failed' also triggers exit 1."""
    _write(tmp_path, RESULT_PATHS[3], {"status": "success", "test_results": "1 failed"})
    assert _run(tmp_path).returncode == 1


def test_no_failed_word_in_test_results_exits_zero(tmp_path):
    _write(tmp_path, RESULT_PATHS[0], {"status": "success", "test_results": "10 passed, 0 skipped"})
    assert _run(tmp_path).returncode == 0


# ---------------------------------------------------------------------------
# Malformed JSON → exit 1
# ---------------------------------------------------------------------------

def test_malformed_json_exits_one(tmp_path):
    p = tmp_path / RESULT_PATHS[0]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not valid json!!!", encoding="utf-8")
    assert _run(tmp_path).returncode == 1


def test_malformed_json_failure_message_contains_path(tmp_path):
    p = tmp_path / RESULT_PATHS[0]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{bad}", encoding="utf-8")
    result = _run(tmp_path)
    assert RESULT_PATHS[0] in result.stdout or RESULT_PATHS[0] in result.stderr


# ---------------------------------------------------------------------------
# Missing files are silently skipped
# ---------------------------------------------------------------------------

def test_missing_files_are_not_treated_as_failure(tmp_path):
    """Only one file present (passing); others absent — must exit 0."""
    _write(tmp_path, RESULT_PATHS[0], {"status": "success"})
    assert _run(tmp_path).returncode == 0


def test_empty_status_field_is_not_failed(tmp_path):
    """A file with status='' should not trip the failure gate."""
    _write(tmp_path, RESULT_PATHS[4], {"status": "", "test_results": ""})
    assert _run(tmp_path).returncode == 0
