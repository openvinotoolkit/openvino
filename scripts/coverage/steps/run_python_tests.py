# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path
import shlex

from coverage_workflow import CoverageContext, env_from_assignments, load_python_tests, run_cmd, warn


PYCOV_CONFIG = """[run]
omit =
    */tests/*
    */thirdparty/*
    */docs/*
    */samples/*
    */tools/*
    */src/bindings/js/node/tests/*
    */src/bindings/python/tests/*
    *.pb.cc
    *.pb.h
"""


def _expand(value: str) -> str:
    return os.path.expandvars(value)


def _find_openvino_wheel(wheels_dir: Path) -> Path:
    candidates = sorted(wheels_dir.glob("openvino-*.whl"))
    if not candidates:
        raise FileNotFoundError(f"OpenVINO wheel not found in {wheels_dir}")
    return candidates[0]


def _run_pytest(test_name: str, target: str, args: str, env_assignments: str) -> int:
    cmd = ["python3", "-m", "pytest", "-ra", "--durations=50", target]
    if args:
        cmd.extend(shlex.split(args))
    env = env_from_assignments(env_assignments)

    print(f"========== Running Python test: {test_name} ==========")
    return run_cmd(cmd, env=env, check=False)


def _run_python_command(test_name: str, command: str, env_assignments: str) -> int:
    merged = command.strip()
    if env_assignments:
        merged = f"{env_assignments} {merged}"

    print(f"========== Running Python test: {test_name} ==========")
    return run_cmd(["bash", "-lc", merged], check=False)


def run(ctx: CoverageContext) -> None:
    config = ctx.workspace / "scripts" / "coverage" / "config" / "tests_python.yml"
    tests = load_python_tests(config, ctx.test_profile)

    if not any(test.enabled for test in tests):
        skipped = [f"{test.name} ({test.skip_reason})" for test in tests]
        ctx.io.export_env("PY_TESTS_TOTAL", str(len(skipped)))
        ctx.io.export_env("PY_TESTS_PASSED", "0")
        ctx.io.export_env("PY_TESTS_FAILED", "0")
        ctx.io.export_env("PY_TESTS_SKIPPED", str(len(skipped)))

        lines = [
            "",
            "## Python coverage test execution summary",
            "Python tests executed: 0",
            "Python tests passed: 0",
            "Python tests failed: 0",
            f"Python tests skipped: {len(skipped)}",
            "",
            "Failed tests: none",
            "",
        ]
        if skipped:
            lines.append("Skipped tests:")
            lines.extend(f"- {item}" for item in skipped)
        else:
            lines.append("Skipped tests: none")
        ctx.io.append_summary("\n".join(lines) + "\n")
        warn(f"No Python tests are enabled for TEST_PROFILE={ctx.test_profile}; skipping Python suite.")
        return

    tests_dir = ctx.paths.install_pkg_dir / "tests"
    src_py_tests = ctx.workspace / "src" / "bindings" / "python" / "tests"
    onnx_py_tests = ctx.workspace / "src" / "frontends" / "onnx" / "tests" / "tests_python"
    layer_tests = ctx.workspace / "tests" / "layer_tests"
    py_cov_config = ctx.workspace / ".python_coverage_ci.rc"

    run_cmd(["python3", "-m", "pip", "install", "-r", str(tests_dir / "bindings/python/requirements_test.txt")])
    run_cmd(["python3", "-m", "pip", "install", "-r", str(tests_dir / "layer_tests/requirements.txt")])
    run_cmd(["python3", "-m", "pip", "install", "-r", str(tests_dir / "requirements_onnx")])
    run_cmd(["python3", "-m", "pip", "install", "-r", str(tests_dir / "requirements_jax")])
    wheel = _find_openvino_wheel(ctx.paths.install_pkg_dir / "wheels")
    run_cmd(["python3", "-m", "pip", "install", "--force-reinstall", str(wheel)])

    os.environ["LD_LIBRARY_PATH"] = f"{ctx.paths.bin_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}".rstrip(":")
    os.environ["PYTHONPATH"] = f"{tests_dir / 'python'}:{os.environ.get('PYTHONPATH', '')}".rstrip(":")

    os.environ["TESTS_DIR"] = str(tests_dir)
    os.environ["SRC_PY_TESTS_DIR"] = str(src_py_tests)
    os.environ["ONNX_PY_TESTS_DIR"] = str(onnx_py_tests)
    os.environ["WORKSPACE_LAYER_TESTS_DIR"] = str(layer_tests)
    os.environ["PY_COV_CONFIG"] = str(py_cov_config)
    os.environ["PYTEST_XDIST_WORKERS"] = str(ctx.pytest_workers)

    py_cov_config.write_text(PYCOV_CONFIG, encoding="utf-8")
    run_cmd(["python3", "-m", "coverage", "erase"])

    executed = 0
    failed: list[str] = []
    skipped: list[str] = []

    for test in tests:
        if not test.enabled:
            skipped.append(f"{test.name} ({test.skip_reason})")
            continue

        target = _expand(test.target)
        args = _expand(test.args)
        test_env = _expand(test.env)
        command = _expand(test.command)

        if test.kind == "pytest":
            executed += 1
            rc = _run_pytest(test.name, target, args, test_env)
            if rc != 0:
                failed.append(f"{test.name} (exit {rc})")
        elif test.kind == "pytest_if_dir":
            if not Path(target).is_dir():
                warn(f"Skipping Python test group '{test.name}' (missing: {target})")
                skipped.append(f"{test.name} (missing path)")
                continue
            executed += 1
            rc = _run_pytest(test.name, target, args, test_env)
            if rc != 0:
                failed.append(f"{test.name} (exit {rc})")
        elif test.kind == "command":
            executed += 1
            rc = _run_python_command(test.name, command, test_env)
            if rc != 0:
                failed.append(f"{test.name} (exit {rc})")
        else:
            warn(f"Unknown Python test kind '{test.kind}' for '{test.name}', skipping")
            skipped.append(f"{test.name} (unknown kind: {test.kind})")

    run_cmd(["python3", "-m", "coverage", "xml", "-o", str(ctx.workspace / "python-coverage.xml")])

    total_failed = len(failed)
    total_skipped = len(skipped)
    total_passed = executed - total_failed

    ctx.io.export_env("PY_TESTS_TOTAL", str(executed + total_skipped))
    ctx.io.export_env("PY_TESTS_PASSED", str(total_passed))
    ctx.io.export_env("PY_TESTS_FAILED", str(total_failed))
    ctx.io.export_env("PY_TESTS_SKIPPED", str(total_skipped))

    lines = [
        "",
        "## Python coverage test execution summary",
        f"Python tests executed: {executed}",
        f"Python tests passed: {total_passed}",
        f"Python tests failed: {total_failed}",
        f"Python tests skipped: {total_skipped}",
        "",
    ]

    if failed:
        lines.append("Failed tests:")
        lines.extend(f"- {item}" for item in failed)
    else:
        lines.append("Failed tests: none")

    lines.append("")

    if skipped:
        lines.append("Skipped tests:")
        lines.extend(f"- {item}" for item in skipped)
    else:
        lines.append("Skipped tests: none")

    ctx.io.append_summary("\n".join(lines) + "\n")

    if failed:
        warn("One or more Python tests failed; continuing to coverage generation.")
