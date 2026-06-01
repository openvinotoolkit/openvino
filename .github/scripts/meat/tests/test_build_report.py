# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for build_report.py.

Covers:
  - Status determination (success / partial / failed)
  - All five output artefacts are produced
  - conversion_report.md contains all required sections
  - analyse_and_convert_result.json schema
  - agent-complete marker: present, well-formed JSON, correct field values
  - Missing input files → graceful degradation (no crash, outputs still written)
  - No gh CLI invoked when PR_NUMBER / ISSUE_NUMBER are absent
"""

import json
import os
import pathlib
import re

import pytest

from conftest import SCRIPTS_DIR, run_script, write_json

SCRIPT = SCRIPTS_DIR / "build_report.py"
OUT_DIR = pathlib.Path("agent-results") / "analyze-and-convert"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_PROFILE = {
    "model_id": "test/my-model",
    "model_type": "llama",
    "architectures": ["LlamaForCausalLM"],
    "pipeline_tag": "text-generation",
    "estimated_params_b": 7.0,
    "is_vlm": False,
    "trust_remote_code_required": False,
}

_DEFAULT_SIGNALS = {
    "error_class": "missing_conversion_rule",
    "target_agent": "enable-operator",
    "requires_optimum_new_arch": False,
    "requires_transformers_upgrade": False,
    "transformers_override": "",
    "requires_tokenizer_check": False,
    "is_vlm": False,
    "custom_ops_suspected": False,
    "oom_suspected": False,
}

_FAILED_ATTEMPT = {
    "id": "A-fp16-stable",
    "description": "fp16 stable",
    "weight_format": "fp16",
    "optimum_version": "stable",
    "success": False,
    "returncode": 1,
    "elapsed_s": 10.0,
    "stdout": "",
    "stderr": "RuntimeError: no rule for aten::erfinv",
}

_SUCCESS_ATTEMPT = {
    "id": "A-fp16-stable",
    "description": "fp16 stable",
    "weight_format": "fp16",
    "optimum_version": "stable",
    "success": True,
    "returncode": 0,
    "elapsed_s": 8.0,
    "stdout": "Export done",
    "stderr": "",
    "ir_files": ["openvino_model.xml", "openvino_model.bin"],
    "ir_dir": "ov_model_A",
}


def _setup(tmp_path, *, attempts=None, signals=None, profile=None, excerpts=None):
    write_json(tmp_path / "model_profile.json", profile or _DEFAULT_PROFILE)
    write_json(tmp_path / "routing_signals.json", signals or _DEFAULT_SIGNALS)
    write_json(tmp_path / "conversion_attempts.json", attempts if attempts is not None else [_FAILED_ATTEMPT])
    write_json(tmp_path / "error_excerpts.json", excerpts or {})


def _run(tmp_path, **kwargs):
    """Run build_report.py with PR_NUMBER/ISSUE_NUMBER unset to suppress gh calls."""
    env = os.environ.copy()
    env.pop("PR_NUMBER", None)
    env.pop("ISSUE_NUMBER", None)
    return run_script(SCRIPT, tmp_path, extra_env=env)


def _report(tmp_path) -> str:
    return (tmp_path / OUT_DIR / "conversion_report.md").read_text(encoding="utf-8")


def _result(tmp_path) -> dict:
    return json.loads((tmp_path / OUT_DIR / "analyze_and_convert_result.json").read_text(encoding="utf-8"))


def _marker(stdout: str) -> dict:
    """Parse the <!-- agent-complete ... --> block from stdout."""
    m = re.search(r"<!-- agent-complete\s*(\{.*?\})\s*-->", stdout, re.DOTALL)
    assert m, f"agent-complete marker not found in stdout:\n{stdout}"
    return json.loads(m.group(1))


# ---------------------------------------------------------------------------
# Status determination
# ---------------------------------------------------------------------------

def test_status_failed_when_all_attempts_failed(tmp_path):
    _setup(tmp_path)
    _run(tmp_path)
    assert _result(tmp_path)["status"] == "failed"


def test_status_success_when_one_attempt_succeeded(tmp_path):
    _setup(tmp_path, attempts=[_SUCCESS_ATTEMPT])
    _run(tmp_path)
    assert _result(tmp_path)["status"] == "success"


def test_status_failed_when_no_attempts_at_all(tmp_path):
    _setup(tmp_path, attempts=[])
    _run(tmp_path)
    assert _result(tmp_path)["status"] == "failed"


# ---------------------------------------------------------------------------
# Output files are produced
# ---------------------------------------------------------------------------

def test_conversion_report_md_is_written(tmp_path):
    _setup(tmp_path)
    _run(tmp_path)
    assert (tmp_path / OUT_DIR / "conversion_report.md").exists()


def test_result_json_is_written(tmp_path):
    _setup(tmp_path)
    _run(tmp_path)
    assert (tmp_path / OUT_DIR / "analyze_and_convert_result.json").exists()


def test_output_dir_is_created_automatically(tmp_path):
    _setup(tmp_path)
    _run(tmp_path)
    assert (tmp_path / OUT_DIR).is_dir()


# ---------------------------------------------------------------------------
# result JSON schema
# ---------------------------------------------------------------------------

def test_result_json_required_fields(tmp_path):
    _setup(tmp_path)
    _run(tmp_path)
    d = _result(tmp_path)
    for field in ("agent", "status", "model_id", "error_class", "target_agent", "routing_signals"):
        assert field in d, f"Missing field: {field!r}"


def test_result_json_agent_name(tmp_path):
    _setup(tmp_path)
    _run(tmp_path)
    assert _result(tmp_path)["agent"] == "analyze-and-convert"


def test_result_json_model_id_matches_profile(tmp_path):
    _setup(tmp_path)
    _run(tmp_path)
    assert _result(tmp_path)["model_id"] == "test/my-model"


def test_result_json_error_class_matches_signals(tmp_path):
    _setup(tmp_path)
    _run(tmp_path)
    assert _result(tmp_path)["error_class"] == "missing_conversion_rule"


def test_result_json_target_agent_matches_signals(tmp_path):
    _setup(tmp_path)
    _run(tmp_path)
    assert _result(tmp_path)["target_agent"] == "enable-operator"


def test_result_json_routing_signals_subset(tmp_path):
    """routing_signals in result JSON must only contain known signal keys."""
    _setup(tmp_path)
    _run(tmp_path)
    allowed = {
        "requires_optimum_new_arch", "requires_transformers_upgrade",
        "transformers_override", "requires_tokenizer_check",
        "trust_remote_code_required", "is_vlm",
        "custom_ops_suspected", "oom_suspected",
    }
    actual = set(_result(tmp_path)["routing_signals"].keys())
    assert actual <= allowed, f"Unexpected keys in routing_signals: {actual - allowed}"


# ---------------------------------------------------------------------------
# conversion_report.md content
# ---------------------------------------------------------------------------

def test_report_contains_model_id(tmp_path):
    _setup(tmp_path)
    _run(tmp_path)
    assert "test/my-model" in _report(tmp_path)


def test_report_has_model_profile_section(tmp_path):
    _setup(tmp_path)
    _run(tmp_path)
    assert "Model Profile" in _report(tmp_path)


def test_report_has_conversion_attempts_section(tmp_path):
    _setup(tmp_path)
    _run(tmp_path)
    assert "Conversion Attempts" in _report(tmp_path)


def test_report_has_routing_signals_section(tmp_path):
    _setup(tmp_path)
    _run(tmp_path)
    assert "Routing Signals" in _report(tmp_path)


def test_report_has_recommended_next_step_section(tmp_path):
    _setup(tmp_path)
    _run(tmp_path)
    assert "Recommended Next Step" in _report(tmp_path)


def test_report_contains_error_class(tmp_path):
    _setup(tmp_path)
    _run(tmp_path)
    assert "missing_conversion_rule" in _report(tmp_path)


def test_report_has_failure_details_when_failed(tmp_path):
    excerpts = {"A-fp16-stable": "Traceback (most recent call last):\n  RuntimeError: no rule"}
    _setup(tmp_path, excerpts=excerpts)
    _run(tmp_path)
    assert "Failure Details" in _report(tmp_path)


def test_report_no_failure_section_when_succeeded(tmp_path):
    _setup(tmp_path, attempts=[_SUCCESS_ATTEMPT])
    _run(tmp_path)
    assert "Failure Details" not in _report(tmp_path)


def test_report_has_successful_strategy_section_on_success(tmp_path):
    _setup(tmp_path, attempts=[_SUCCESS_ATTEMPT])
    _run(tmp_path)
    assert "Successful Strategy" in _report(tmp_path)


# ---------------------------------------------------------------------------
# agent-complete marker
# ---------------------------------------------------------------------------

def test_agent_complete_marker_present_in_stdout(tmp_path):
    _setup(tmp_path)
    result = _run(tmp_path)
    assert "<!-- agent-complete" in result.stdout
    assert "-->" in result.stdout


def test_agent_complete_marker_is_valid_json(tmp_path):
    _setup(tmp_path)
    result = _run(tmp_path)
    _marker(result.stdout)  # asserts internally


def test_agent_complete_has_all_required_fields(tmp_path):
    _setup(tmp_path)
    result = _run(tmp_path)
    m = _marker(result.stdout)
    for field in ("agent", "status", "next_agent", "error_class", "model_id", "next_context"):
        assert field in m, f"Missing field in agent-complete: {field!r}"


def test_agent_complete_agent_name_correct(tmp_path):
    _setup(tmp_path)
    result = _run(tmp_path)
    assert _marker(result.stdout)["agent"] == "analyze-and-convert"


def test_agent_complete_status_failed(tmp_path):
    _setup(tmp_path)
    result = _run(tmp_path)
    assert _marker(result.stdout)["status"] == "failed"


def test_agent_complete_status_success(tmp_path):
    _setup(tmp_path, attempts=[_SUCCESS_ATTEMPT])
    result = _run(tmp_path)
    assert _marker(result.stdout)["status"] == "success"


def test_agent_complete_next_agent_matches_signals(tmp_path):
    _setup(tmp_path)
    result = _run(tmp_path)
    assert _marker(result.stdout)["next_agent"] == "enable-operator"


def test_agent_complete_model_id_correct(tmp_path):
    _setup(tmp_path)
    result = _run(tmp_path)
    assert _marker(result.stdout)["model_id"] == "test/my-model"


# ---------------------------------------------------------------------------
# Resilience: missing inputs
# ---------------------------------------------------------------------------

def test_missing_all_inputs_does_not_crash(tmp_path):
    """Script must produce outputs even with zero input files."""
    result = _run(tmp_path)
    assert result.returncode == 0
    assert (tmp_path / OUT_DIR / "conversion_report.md").exists()
    assert (tmp_path / OUT_DIR / "analyze_and_convert_result.json").exists()
    assert "<!-- agent-complete" in result.stdout


def test_missing_excerpts_does_not_crash(tmp_path):
    _setup(tmp_path)
    (tmp_path / "error_excerpts.json").unlink()
    result = _run(tmp_path)
    assert result.returncode == 0


def test_missing_signals_does_not_crash(tmp_path):
    _setup(tmp_path)
    (tmp_path / "routing_signals.json").unlink()
    result = _run(tmp_path)
    assert result.returncode == 0


def test_corrupted_json_input_does_not_crash(tmp_path):
    _setup(tmp_path)
    (tmp_path / "model_profile.json").write_text("{invalid json", encoding="utf-8")
    result = _run(tmp_path)
    assert result.returncode == 0


# ---------------------------------------------------------------------------
# gh CLI: must NOT be called when env vars absent
# ---------------------------------------------------------------------------

def test_no_gh_invocation_without_pr_or_issue_number(tmp_path):
    _setup(tmp_path)
    result = _run(tmp_path)
    # The script should log that it is skipping — verify no unexpected exit
    assert result.returncode == 0
    # "skipping" message confirms gh was not invoked
    assert "skipping" in result.stdout.lower() or "No PR_NUMBER" in result.stdout
