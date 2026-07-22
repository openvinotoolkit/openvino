# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for classify_component.py.

Reads agent-results/pipeline_state.json and maps error_context to one of:
  frontend | core_op | cpu_plugin | transformation

Safe defaults:
  - missing state file  → component=frontend (exit 0)
  - unknown error class → component=frontend (exit 0)

co_located_ops in state → printed to stdout (routing hint, non-fatal).
"""

import json

import pytest

from conftest import SCRIPTS_DIR, run_script, write_json

SCRIPT = SCRIPTS_DIR / "classify_component.py"
STATE = "agent-results/pipeline_state.json"


def _run(tmp_path):
    return run_script(SCRIPT, tmp_path)


def _write_state(tmp_path, error_context="", co_located_ops=None):
    write_json(tmp_path / STATE, {
        "ov_orchestrator": {
            "error_context": error_context,
            "co_located_ops": co_located_ops or [],
        }
    })


# ---------------------------------------------------------------------------
# Missing state file → safe default
# ---------------------------------------------------------------------------

def test_missing_state_file_exits_zero(tmp_path):
    assert _run(tmp_path).returncode == 0


def test_missing_state_file_outputs_frontend(tmp_path):
    assert "component=frontend" in _run(tmp_path).stdout


# ---------------------------------------------------------------------------
# Classification map — all five entries
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("error_class, expected_component", [
    ("missing_conversion_rule",  "frontend"),
    ("frontend_error",           "frontend"),
    ("ir_validation_error",      "core_op"),
    ("inference_runtime_error",  "cpu_plugin"),
    ("accuracy_regression",      "transformation"),
])
def test_classification_map_entry(tmp_path, error_class, expected_component):
    _write_state(tmp_path, error_context=error_class)
    result = _run(tmp_path)
    assert result.returncode == 0
    assert f"component={expected_component}" in result.stdout


# ---------------------------------------------------------------------------
# Unknown / empty error class → fallback to frontend
# ---------------------------------------------------------------------------

def test_unknown_error_class_defaults_to_frontend(tmp_path):
    _write_state(tmp_path, error_context="xyzzy_completely_unknown")
    assert "component=frontend" in _run(tmp_path).stdout


def test_empty_error_context_defaults_to_frontend(tmp_path):
    _write_state(tmp_path, error_context="")
    assert "component=frontend" in _run(tmp_path).stdout


# ---------------------------------------------------------------------------
# error_context with a slash prefix (sub-path format)
# ---------------------------------------------------------------------------

def test_slash_separated_context_uses_prefix_only(tmp_path):
    """'ir_validation_error/some/detail' must resolve to core_op via prefix."""
    _write_state(tmp_path, error_context="ir_validation_error/some/extra/detail")
    assert "component=core_op" in _run(tmp_path).stdout


def test_slash_separated_frontend_error(tmp_path):
    _write_state(tmp_path, error_context="missing_conversion_rule/aten::erfinv")
    assert "component=frontend" in _run(tmp_path).stdout


# ---------------------------------------------------------------------------
# co_located_ops — printed but not a failure
# ---------------------------------------------------------------------------

def test_co_located_ops_printed_in_stdout(tmp_path):
    _write_state(tmp_path, error_context="frontend_error",
                 co_located_ops=["aten::erfinv", "aten::logit"])
    result = _run(tmp_path)
    assert "aten::erfinv" in result.stdout
    assert "co_located_ops" in result.stdout


def test_co_located_ops_does_not_change_exit_code(tmp_path):
    _write_state(tmp_path, error_context="frontend_error",
                 co_located_ops=["aten::erfinv"])
    assert _run(tmp_path).returncode == 0


def test_empty_co_located_ops_no_multi_op_message(tmp_path):
    _write_state(tmp_path, error_context="frontend_error", co_located_ops=[])
    assert "Multi-op" not in _run(tmp_path).stdout


# ---------------------------------------------------------------------------
# Malformed state file → should not crash (safe fallback)
# ---------------------------------------------------------------------------

def test_malformed_state_json_handled_gracefully(tmp_path):
    p = tmp_path / STATE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{invalid json", encoding="utf-8")
    # Must not crash with an unhandled exception (returncode != 2 which is Python traceback)
    result = _run(tmp_path)
    assert result.returncode != 2, f"Unhandled exception:\n{result.stderr}"
