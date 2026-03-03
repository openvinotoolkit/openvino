# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .models import CppTestCase, JsTestCase, PythonTestCase, SUPPORTED_PROFILES


@dataclass(frozen=True)
class ConfigValidationIssue:
    suite: str
    test_name: str
    message: str


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _resolve_profile_value(value: Any, profile: str) -> str:
    if isinstance(value, dict):
        if profile in value:
            return _as_text(value[profile])
        return _as_text(value.get("default", ""))
    return _as_text(value)


def _resolve_enabled(test: dict[str, Any], profile: str) -> tuple[bool, str]:
    skip_reason = _as_text(test.get("skip_reason", "")).strip()
    profiles = test.get("profiles")

    if profiles is not None:
        normalized = [str(p) for p in profiles]
        if profile not in normalized:
            reason = _as_text(test.get("profile_skip_reason", "")).strip()
            if not reason:
                reason = f"profile '{profile}' is OFF"
            return False, reason

    if skip_reason:
        return False, skip_reason

    return True, ""


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML root in {path}")
    return data


def _load_tests(path: Path, suite: str) -> list[dict[str, Any]]:
    data = _load_yaml(path)
    found_suite = _as_text(data.get("suite", "")).strip()
    if found_suite != suite:
        raise ValueError(f"Invalid suite in {path}: expected '{suite}', got '{found_suite}'")
    tests = data.get("tests")
    if not isinstance(tests, list):
        raise ValueError(f"Invalid config: 'tests' list is missing in {path}")
    return tests


def load_cpp_tests(path: Path, profile: str) -> list[CppTestCase]:
    if profile not in SUPPORTED_PROFILES:
        raise ValueError(f"Unsupported profile: {profile}")

    loaded: list[CppTestCase] = []
    for test in _load_tests(path, "cpp"):
        enabled, reason = _resolve_enabled(test, profile)
        loaded.append(
            CppTestCase(
                name=_as_text(test.get("name", "")).strip(),
                enabled=enabled,
                skip_reason=reason,
                binary=_as_text(test.get("binary", "")).strip(),
                mode=_as_text(test.get("mode", "gtest_single")).strip() or "gtest_single",
                args=_resolve_profile_value(test.get("args", ""), profile).strip(),
                extra_env=_resolve_profile_value(test.get("extra_env", ""), profile).strip(),
            )
        )
    return loaded


def load_python_tests(path: Path, profile: str) -> list[PythonTestCase]:
    if profile not in SUPPORTED_PROFILES:
        raise ValueError(f"Unsupported profile: {profile}")

    loaded: list[PythonTestCase] = []
    for test in _load_tests(path, "python"):
        enabled, reason = _resolve_enabled(test, profile)
        loaded.append(
            PythonTestCase(
                name=_as_text(test.get("name", "")).strip(),
                enabled=enabled,
                skip_reason=reason,
                kind=_as_text(test.get("kind", "pytest")).strip() or "pytest",
                target=_resolve_profile_value(test.get("target", ""), profile).strip(),
                args=_resolve_profile_value(test.get("args", ""), profile).strip(),
                env=_resolve_profile_value(test.get("env", ""), profile).strip(),
                command=_resolve_profile_value(test.get("command", ""), profile).strip(),
            )
        )
    return loaded


def load_js_tests(path: Path, profile: str) -> list[JsTestCase]:
    if profile not in SUPPORTED_PROFILES:
        raise ValueError(f"Unsupported profile: {profile}")

    loaded: list[JsTestCase] = []
    for test in _load_tests(path, "js"):
        enabled, reason = _resolve_enabled(test, profile)
        loaded.append(
            JsTestCase(
                name=_as_text(test.get("name", "")).strip(),
                enabled=enabled,
                skip_reason=reason,
                kind=_as_text(test.get("kind", "command")).strip() or "command",
                command=_resolve_profile_value(test.get("command", ""), profile).strip(),
            )
        )
    return loaded


def validate_configs(config_dir: Path) -> list[ConfigValidationIssue]:
    issues: list[ConfigValidationIssue] = []

    suites = {
        "cpp": config_dir / "tests_cpp.yml",
        "python": config_dir / "tests_python.yml",
        "js": config_dir / "tests_js.yml",
    }

    for suite, path in suites.items():
        tests = _load_tests(path, suite)
        for idx, test in enumerate(tests):
            name = _as_text(test.get("name", f"<index:{idx}>"))
            if not _as_text(test.get("name", "")).strip():
                issues.append(ConfigValidationIssue(suite, name, "missing 'name'"))
            if suite == "cpp" and not _as_text(test.get("binary", "")).strip():
                issues.append(ConfigValidationIssue(suite, name, "missing 'binary'"))
            if suite == "python":
                kind = _as_text(test.get("kind", "pytest")).strip() or "pytest"
                if kind not in {"pytest", "pytest_if_dir", "command"}:
                    issues.append(ConfigValidationIssue(suite, name, f"unsupported kind '{kind}'"))
            if suite == "js":
                kind = _as_text(test.get("kind", "command")).strip() or "command"
                if kind != "command":
                    issues.append(ConfigValidationIssue(suite, name, f"unsupported kind '{kind}'"))

    return issues
