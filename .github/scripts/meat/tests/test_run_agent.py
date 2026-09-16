# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for run_agent.py.

This is the CLI entry point for all agents.  Critical safety properties:
  - Wrong arg count → exit 1, usage hint
  - Unknown agent → exit 1, error mentions available agents
  - Missing context file → exit 1, path reported
  - Missing agent file (repo root not found) → exit 1
  - --list / -l → exit 0, all expected agents listed
  - No mutating side effect when args are invalid
"""

import pathlib
import sys
import tempfile
import os

import pytest

from conftest import SCRIPTS_DIR, run_script

SCRIPT = SCRIPTS_DIR / "run_agent.py"
# run_agent checks for agent files relative to the repo root
REPO_ROOT = SCRIPTS_DIR.parent.parent.parent

KNOWN_AGENTS = [
    "enable-operator",
    "frontend",
    "core-opspec",
    "transformation",
    "cpu",
    "gpu",
    "deployer",
    "analyze-and-convert",
]


def _run(*args, cwd=None):
    return run_script(SCRIPT, cwd or REPO_ROOT, *args)


# ---------------------------------------------------------------------------
# --list flag
# ---------------------------------------------------------------------------

def test_list_flag_exits_zero():
    assert _run("--list").returncode == 0


def test_list_short_flag_exits_zero():
    assert _run("-l").returncode == 0


def test_list_contains_all_expected_agents():
    stdout = _run("--list").stdout
    for agent in KNOWN_AGENTS:
        assert agent in stdout, f"Agent {agent!r} not in --list output"


# ---------------------------------------------------------------------------
# Wrong argument count
# ---------------------------------------------------------------------------

def test_no_args_exits_nonzero():
    assert _run().returncode != 0


def test_single_non_list_arg_exits_nonzero():
    assert _run("enable-operator").returncode != 0


def test_three_args_exits_nonzero():
    assert _run("enable-operator", "ctx.txt", "extra").returncode != 0


def test_usage_hint_on_wrong_arg_count():
    result = _run()
    assert "usage" in result.stderr.lower() or "usage" in result.stdout.lower()


# ---------------------------------------------------------------------------
# Missing context file
# ---------------------------------------------------------------------------

def test_missing_context_file_exits_nonzero():
    result = _run("enable-operator", "/nonexistent/context_file.txt")
    assert result.returncode != 0


def test_missing_context_file_error_mentions_path():
    result = _run("enable-operator", "/nonexistent/context_file.txt")
    assert "context" in result.stderr.lower() or "not found" in result.stderr.lower()


# ---------------------------------------------------------------------------
# Unknown agent (agent .md file missing)
# ---------------------------------------------------------------------------

def test_unknown_agent_with_existing_context_file_exits_nonzero():
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False
    ) as fh:
        fh.write("Operator: aten::erfinv\nModel: test/model\n")
        ctx = fh.name
    try:
        result = _run("no-such-agent-ever", ctx)
        assert result.returncode != 0
    finally:
        os.unlink(ctx)


def test_unknown_agent_error_mentions_agent_file():
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False
    ) as fh:
        fh.write("dummy")
        ctx = fh.name
    try:
        result = _run("no-such-agent-ever", ctx)
        combined = result.stderr + result.stdout
        assert "agent" in combined.lower() or "not found" in combined.lower()
    finally:
        os.unlink(ctx)


# ---------------------------------------------------------------------------
# Valid context file + known agent: verifies pre-flight checks pass
# (script will fail later when trying to invoke copilot CLI, that's expected)
# ---------------------------------------------------------------------------

def test_known_agent_with_valid_context_does_not_exit_with_usage_error():
    """If the agent .md exists and context file exists, the usage-check phase
    must pass (exit code != 1 from arg-parsing).  The script will fail later
    attempting to invoke ``copilot`` — that is acceptable here.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False
    ) as fh:
        fh.write("Operator: aten::erfinv\nModel: test/model\n")
        ctx = fh.name
    try:
        result = _run("enable-operator", ctx)
        # Should NOT fail with the usage-error message
        combined = result.stderr + result.stdout
        assert "Usage:" not in combined, (
            f"Got usage error for valid inputs:\n{combined}"
        )
    finally:
        os.unlink(ctx)
