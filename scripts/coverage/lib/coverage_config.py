#!/usr/bin/env python3

# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import sys

try:
    import yaml
except ImportError as exc:
    sys.stderr.write("[coverage][error] PyYAML is required (pip install pyyaml).\n")
    raise SystemExit(2) from exc

SUPPORTED_PROFILES = {"cpu", "cpu_gpu", "cpu_npu", "cpu_npu_gpu"}


def _as_text(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _clean(text):
    return _as_text(text).replace("\t", " ").replace("\n", " ").strip()


def _resolve_profile_value(value, profile):
    if isinstance(value, dict):
        if profile in value:
            return _as_text(value[profile])
        return _as_text(value.get("default", ""))
    return _as_text(value)


def _resolve_enabled(test, profile):
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


def _load_tests(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    tests = data.get("tests")
    if not isinstance(tests, list):
        raise ValueError(f"Invalid config: 'tests' list is missing in {config_path}")

    return tests


def emit_cpp(config_path, profile):
    tests = _load_tests(config_path)

    for test in tests:
        enabled, reason = _resolve_enabled(test, profile)
        name = _clean(test.get("name", ""))
        binary = _clean(test.get("binary", ""))
        mode = _clean(test.get("mode", "gtest_single")) or "gtest_single"
        args = _clean(_resolve_profile_value(test.get("args", ""), profile))
        extra_env = _clean(_resolve_profile_value(test.get("extra_env", ""), profile))

        row = [
            name,
            "1" if enabled else "0",
            _clean(reason),
            binary,
            mode,
            args,
            extra_env,
        ]
        print("\t".join(row))


def emit_python(config_path, profile):
    tests = _load_tests(config_path)

    for test in tests:
        enabled, reason = _resolve_enabled(test, profile)
        name = _clean(test.get("name", ""))
        kind = _clean(test.get("kind", "pytest"))
        target = _clean(_resolve_profile_value(test.get("target", ""), profile))
        args = _clean(_resolve_profile_value(test.get("args", ""), profile))
        env = _clean(_resolve_profile_value(test.get("env", ""), profile))
        command = _clean(_resolve_profile_value(test.get("command", ""), profile))

        row = [
            name,
            "1" if enabled else "0",
            _clean(reason),
            kind,
            target,
            args,
            env,
            command,
        ]
        print("\t".join(row))


def emit_js(config_path, profile):
    tests = _load_tests(config_path)

    for test in tests:
        enabled, reason = _resolve_enabled(test, profile)
        name = _clean(test.get("name", ""))
        kind = _clean(test.get("kind", "command")) or "command"
        command = _clean(_resolve_profile_value(test.get("command", ""), profile))

        row = [
            name,
            "1" if enabled else "0",
            _clean(reason),
            kind,
            command,
        ]
        print("\t".join(row))


def main():
    parser = argparse.ArgumentParser(description="Emit coverage test config rows for Bash runners")
    parser.add_argument("--suite", required=True, choices=["cpp", "python", "js"])
    parser.add_argument("--profile", required=True, help="Coverage profile")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    profile = args.profile.strip()
    if profile not in SUPPORTED_PROFILES:
        raise SystemExit(f"Unsupported profile: {profile}")

    config_path = os.path.abspath(args.config)
    if not os.path.isfile(config_path):
        raise SystemExit(f"Config file not found: {config_path}")

    if args.suite == "cpp":
        emit_cpp(config_path, profile)
    elif args.suite == "python":
        emit_python(config_path, profile)
    else:
        emit_js(config_path, profile)


if __name__ == "__main__":
    main()
