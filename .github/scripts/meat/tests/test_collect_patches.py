# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for collect_patches.py.

Reads sub-agent result JSON files, copies referenced patch files to
  agent-results/enable-operator/patches/openvino/
and produces a combined patch file.

Exit codes:
  0 — all referenced patches found (or zero patches total)
  1 — at least one referenced patch file is missing

Resilience requirements:
  - Missing result JSON files → silently skipped
  - Malformed result JSON → silently skipped
  - Missing patch file → exit 1 + warning to stderr
  - Both patch_paths (list) and patch_path (singular) keys supported
"""

import json
import pathlib

import pytest

from conftest import SCRIPTS_DIR, run_script, write_json

SCRIPT = SCRIPTS_DIR / "collect_patches.py"

RESULT_FILES = [
    "agent-results/frontend/fe_result.json",
    "agent-results/core-opspec/core_opspec_result.json",
    "agent-results/transformation/transformation_result.json",
    "agent-results/cpu/cpu_result.json",
    "agent-results/gpu/gpu_result.json",
    "agent-results/npu/npu_result.json",
]

OUT_DIR = pathlib.Path("agent-results/enable-operator/patches/openvino")
COMBINED = pathlib.Path("agent-results/enable-operator/patches/openvino_combined.patch")


def _run(tmp_path):
    return run_script(SCRIPT, tmp_path)


def _result(tmp_path, rel_path, data):
    write_json(tmp_path / rel_path, data)


def _patch(tmp_path, name, content="diff --git a/x b/x\n+added line\n"):
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return str(p)


# ---------------------------------------------------------------------------
# No result files → exit 0, "No patches" message
# ---------------------------------------------------------------------------

def test_no_result_files_exits_zero(tmp_path):
    assert _run(tmp_path).returncode == 0


def test_no_result_files_warns_no_patches(tmp_path):
    assert "no patches" in _run(tmp_path).stdout.lower()


def test_output_dir_created_even_when_no_patches(tmp_path):
    _run(tmp_path)
    assert (tmp_path / OUT_DIR).is_dir()


# ---------------------------------------------------------------------------
# Successful patch collection
# ---------------------------------------------------------------------------

def test_patch_is_copied_to_output_dir(tmp_path):
    pp = _patch(tmp_path, "frontend.patch", "diff frontend content")
    _result(tmp_path, RESULT_FILES[0], {"status": "success", "patch_paths": [pp]})
    _run(tmp_path)
    assert (tmp_path / OUT_DIR / "frontend.patch").exists()


def test_copied_patch_content_is_intact(tmp_path):
    pp = _patch(tmp_path, "core.patch", "unique-content-XYZZY")
    _result(tmp_path, RESULT_FILES[1], {"patch_paths": [pp]})
    _run(tmp_path)
    content = (tmp_path / OUT_DIR / "core.patch").read_text(encoding="utf-8")
    assert "unique-content-XYZZY" in content


def test_combined_patch_is_created(tmp_path):
    pp = _patch(tmp_path, "a.patch", "content-A")
    _result(tmp_path, RESULT_FILES[0], {"patch_paths": [pp]})
    _run(tmp_path)
    assert (tmp_path / COMBINED).exists()


def test_combined_patch_contains_all_collected_content(tmp_path):
    p1 = _patch(tmp_path, "fe.patch",  "content-FE")
    p2 = _patch(tmp_path, "cpu.patch", "content-CPU")
    _result(tmp_path, RESULT_FILES[0], {"patch_paths": [p1]})
    _result(tmp_path, RESULT_FILES[3], {"patch_paths": [p2]})
    _run(tmp_path)
    combined = (tmp_path / COMBINED).read_text(encoding="utf-8")
    assert "content-FE"  in combined
    assert "content-CPU" in combined


def test_multiple_patches_per_result_all_collected(tmp_path):
    p1 = _patch(tmp_path, "part1.patch", "part-one")
    p2 = _patch(tmp_path, "part2.patch", "part-two")
    _result(tmp_path, RESULT_FILES[0], {"patch_paths": [p1, p2]})
    _run(tmp_path)
    assert (tmp_path / OUT_DIR / "part1.patch").exists()
    assert (tmp_path / OUT_DIR / "part2.patch").exists()


def test_singular_patch_path_key_supported(tmp_path):
    """Older result files may use 'patch_path' (string) not 'patch_paths' (list)."""
    pp = _patch(tmp_path, "singular.patch", "singular-content")
    _result(tmp_path, RESULT_FILES[1], {"patch_path": pp})
    _run(tmp_path)
    assert (tmp_path / OUT_DIR / "singular.patch").exists()


def test_patch_paths_list_takes_precedence_over_singular_key(tmp_path):
    """When patch_paths (list) is present, it is used; patch_path (singular) is a fallback only."""
    p1 = _patch(tmp_path, "list.patch", "list-content")
    _result(tmp_path, RESULT_FILES[0], {"patch_paths": [p1], "patch_path": "/nonexistent/ignored.patch"})
    # patch_path is ignored when patch_paths is present — must not exit 1 for the unreferenced path
    result = _run(tmp_path)
    assert result.returncode == 0
    assert (tmp_path / OUT_DIR / "list.patch").exists()


# ---------------------------------------------------------------------------
# Missing patch file → exit 1
# ---------------------------------------------------------------------------

def test_missing_patch_file_exits_one(tmp_path):
    _result(tmp_path, RESULT_FILES[0], {"patch_paths": ["/nonexistent/ghost.patch"]})
    assert _run(tmp_path).returncode == 1


def test_missing_patch_file_warning_to_stderr(tmp_path):
    _result(tmp_path, RESULT_FILES[0], {"patch_paths": ["/nonexistent/ghost.patch"]})
    result = _run(tmp_path)
    assert result.stderr, "Expected warning on stderr for missing patch"


def test_missing_patch_exits_one_even_if_others_succeed(tmp_path):
    """Partial failure: one found + one missing → still exit 1."""
    good = _patch(tmp_path, "good.patch", "OK")
    _result(tmp_path, RESULT_FILES[0], {"patch_paths": [good]})
    _result(tmp_path, RESULT_FILES[1], {"patch_paths": ["/nonexistent/bad.patch"]})
    assert _run(tmp_path).returncode == 1


# ---------------------------------------------------------------------------
# Resilience
# ---------------------------------------------------------------------------

def test_malformed_json_result_skipped_no_crash(tmp_path):
    p = tmp_path / RESULT_FILES[0]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{invalid json!!!", encoding="utf-8")
    result = _run(tmp_path)
    # Malformed JSON = skip, not crash; exit 0 (no patches found)
    assert result.returncode == 0


def test_missing_result_files_skipped_silently(tmp_path):
    """Only one file present (with a valid patch); others absent → exit 0."""
    pp = _patch(tmp_path, "x.patch", "content")
    _result(tmp_path, RESULT_FILES[4], {"patch_paths": [pp]})
    result = _run(tmp_path)
    assert result.returncode == 0
    assert (tmp_path / OUT_DIR / "x.patch").exists()


def test_result_file_with_empty_patch_paths_skipped(tmp_path):
    _result(tmp_path, RESULT_FILES[0], {"patch_paths": []})
    result = _run(tmp_path)
    assert result.returncode == 0


def test_result_file_with_no_patch_key_skipped(tmp_path):
    """Result files that don't mention patches at all are fine."""
    _result(tmp_path, RESULT_FILES[0], {"status": "success"})
    result = _run(tmp_path)
    assert result.returncode == 0
