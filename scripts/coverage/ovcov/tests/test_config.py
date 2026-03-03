# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from ovcov.config import load_cpp_tests, validate_configs


def _config_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "config"


def test_cpp_filters_cpu_profile() -> None:
    tests = load_cpp_tests(_config_dir() / "tests_cpp.yml", "cpu")
    target = next(t for t in tests if t.name == "ov_core_unit_tests")
    assert target.enabled
    assert target.args == "-*IE_GPU*"


def test_cpp_filters_gpu_profile() -> None:
    tests = load_cpp_tests(_config_dir() / "tests_cpp.yml", "cpu_gpu")
    target = next(t for t in tests if t.name == "ov_core_unit_tests")
    assert target.enabled
    assert target.args == ""


def test_config_validation_has_no_issues() -> None:
    issues = validate_configs(_config_dir())
    assert issues == []
