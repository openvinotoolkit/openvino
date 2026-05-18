# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for try_conversion.py.

Uses a cross-platform ``optimum-cli`` stub injected via PATH to exercise
the strategy-matrix logic without invoking the real tool.

Critical properties under test:
  - Missing model_profile.json → exit 1, clear error
  - Strategy D (int4 AWQ) is present for >7B models, absent for ≤7B
  - Script stops on the first successful strategy
  - Exit 0 without IR files on disk is NOT treated as success
  - All conversion attempts are persisted with required fields
  - Failed attempts have success=False
  - conversion_attempts.json is always written (even on all-failure)
"""

import json
import os
import pathlib
import sys

import pytest

from conftest import SCRIPTS_DIR, run_script, write_json, make_fake_cmd, patched_env

SCRIPT = SCRIPTS_DIR / "try_conversion.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_profile(tmp_path, **overrides):
    profile = {
        "model_id": "test/model",
        "trust_remote_code_required": False,
        "is_vlm": False,
        "optimum_supported": True,
        "estimated_params_b": 1.0,
        "pipeline_tag": "text-generation",
    }
    profile.update(overrides)
    write_json(tmp_path / "model_profile.json", profile)


def _run_with_stub(tmp_path, bin_dir):
    env = patched_env(bin_dir)
    # Skip pip install calls during tests to avoid network access.
    env["_TRY_CONVERSION_SKIP_PIP"] = "1"
    return run_script(SCRIPT, tmp_path, extra_env=env)


def _attempts(tmp_path) -> list:
    p = tmp_path / "conversion_attempts.json"
    assert p.exists(), "conversion_attempts.json was not written"
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Guard: missing model_profile.json
# ---------------------------------------------------------------------------

def test_missing_profile_exits_one(tmp_path):
    result = run_script(SCRIPT, tmp_path)
    assert result.returncode == 1


def test_missing_profile_error_mentions_file(tmp_path):
    result = run_script(SCRIPT, tmp_path)
    assert "model_profile.json" in result.stderr


# ---------------------------------------------------------------------------
# Strategy matrix: int4 AWQ (Strategy D)
# ---------------------------------------------------------------------------

def test_int4_strategy_absent_for_small_model(tmp_path):
    """Models ≤7B must never trigger the expensive int4 AWQ strategy."""
    _write_profile(tmp_path, estimated_params_b=1.0)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    make_fake_cmd(bin_dir, "optimum-cli", exit_code=1)

    _run_with_stub(tmp_path, bin_dir)

    ids = [a["id"] for a in _attempts(tmp_path)]
    assert not any("int4" in i.lower() for i in ids), (
        f"int4 strategy incorrectly present for 1B model: {ids}"
    )


def test_int4_strategy_present_for_large_model(tmp_path):
    """Models >7B must include the int4 AWQ strategy in the matrix."""
    _write_profile(tmp_path, estimated_params_b=70.0)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    make_fake_cmd(bin_dir, "optimum-cli", exit_code=1)

    _run_with_stub(tmp_path, bin_dir)

    ids = [a["id"] for a in _attempts(tmp_path)]
    assert any("int4" in i.lower() for i in ids), (
        f"int4 strategy missing for 70B model: {ids}"
    )


def test_boundary_exactly_7b_does_not_get_int4(tmp_path):
    """Exactly 7.0B is NOT greater than 7 → no int4 strategy."""
    _write_profile(tmp_path, estimated_params_b=7.0)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    make_fake_cmd(bin_dir, "optimum-cli", exit_code=1)

    _run_with_stub(tmp_path, bin_dir)

    ids = [a["id"] for a in _attempts(tmp_path)]
    assert not any("int4" in i.lower() for i in ids)


# ---------------------------------------------------------------------------
# Stop-on-success behaviour
# ---------------------------------------------------------------------------

def test_stops_after_first_successful_strategy(tmp_path):
    """Only strategy A should run when it succeeds immediately."""
    _write_profile(tmp_path)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    make_fake_cmd(bin_dir, "optimum-cli", exit_code=0, create_ir=True)

    _run_with_stub(tmp_path, bin_dir)

    attempts = _attempts(tmp_path)
    assert len(attempts) == 1, (
        f"Expected exactly 1 attempt after first success, got {len(attempts)}"
    )
    assert attempts[0]["success"] is True


def test_all_strategies_run_when_all_fail(tmp_path):
    """All strategies (A-E, or A-E+D for large model) must be tried on failure."""
    _write_profile(tmp_path, estimated_params_b=1.0)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    make_fake_cmd(bin_dir, "optimum-cli", exit_code=1)

    _run_with_stub(tmp_path, bin_dir)

    attempts = _attempts(tmp_path)
    # Small model: A, B, C, E  (no D) = 4 strategies
    assert len(attempts) >= 4, (
        f"Expected ≥4 attempts for 1B model, got {len(attempts)}: {[a['id'] for a in attempts]}"
    )


# ---------------------------------------------------------------------------
# Exit-0-without-IR-files is NOT success
# ---------------------------------------------------------------------------

def test_exit_zero_without_ir_files_is_not_success(tmp_path):
    """optimum-cli returning 0 but producing no .xml/.bin must not be success."""
    _write_profile(tmp_path)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # create_ir=False: exits 0 but writes nothing
    make_fake_cmd(bin_dir, "optimum-cli", exit_code=0, create_ir=False)

    _run_with_stub(tmp_path, bin_dir)

    for a in _attempts(tmp_path):
        assert a["success"] is False, (
            f"Attempt {a['id']} marked success despite no IR files"
        )


# ---------------------------------------------------------------------------
# Output: conversion_attempts.json structure
# ---------------------------------------------------------------------------

def test_conversion_attempts_json_written_on_all_failure(tmp_path):
    _write_profile(tmp_path)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    make_fake_cmd(bin_dir, "optimum-cli", exit_code=1)

    _run_with_stub(tmp_path, bin_dir)
    assert (tmp_path / "conversion_attempts.json").exists()


def test_each_attempt_has_required_fields(tmp_path):
    _write_profile(tmp_path)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    make_fake_cmd(bin_dir, "optimum-cli", exit_code=1)

    _run_with_stub(tmp_path, bin_dir)

    required = {"id", "description", "command", "returncode", "success", "stdout", "stderr"}
    for a in _attempts(tmp_path):
        missing = required - set(a.keys())
        assert not missing, f"Attempt {a.get('id')!r} missing fields: {missing}"


def test_failed_attempts_have_success_false(tmp_path):
    _write_profile(tmp_path)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    make_fake_cmd(bin_dir, "optimum-cli", exit_code=1)

    _run_with_stub(tmp_path, bin_dir)

    for a in _attempts(tmp_path):
        assert a["success"] is False


def test_successful_attempt_recorded_with_ir_files(tmp_path):
    _write_profile(tmp_path)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    make_fake_cmd(bin_dir, "optimum-cli", exit_code=0, create_ir=True)

    _run_with_stub(tmp_path, bin_dir)

    winner = _attempts(tmp_path)[0]
    assert winner["success"] is True
    assert "openvino_model.xml" in winner.get("ir_files", [])


def test_returncode_recorded_in_attempt(tmp_path):
    _write_profile(tmp_path)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    make_fake_cmd(bin_dir, "optimum-cli", exit_code=42)

    _run_with_stub(tmp_path, bin_dir)

    assert _attempts(tmp_path)[0]["returncode"] == 42


def test_command_string_contains_model_id(tmp_path):
    _write_profile(tmp_path, model_id="org/my-special-model")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    make_fake_cmd(bin_dir, "optimum-cli", exit_code=1)

    _run_with_stub(tmp_path, bin_dir)

    for a in _attempts(tmp_path):
        assert "org/my-special-model" in a["command"]


# ---------------------------------------------------------------------------
# trust_remote_code flag forwarded
# ---------------------------------------------------------------------------

def test_trust_remote_code_flag_in_command_when_required(tmp_path):
    _write_profile(tmp_path, trust_remote_code_required=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    make_fake_cmd(bin_dir, "optimum-cli", exit_code=1)

    _run_with_stub(tmp_path, bin_dir)

    for a in _attempts(tmp_path):
        assert "--trust-remote-code" in a["command"], (
            f"--trust-remote-code missing from command: {a['command']}"
        )


def test_no_trust_remote_code_flag_when_not_required(tmp_path):
    _write_profile(tmp_path, trust_remote_code_required=False)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    make_fake_cmd(bin_dir, "optimum-cli", exit_code=1)

    _run_with_stub(tmp_path, bin_dir)

    for a in _attempts(tmp_path):
        assert "--trust-remote-code" not in a["command"], (
            f"--trust-remote-code should not appear: {a['command']}"
        )
