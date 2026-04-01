# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import os
from pathlib import Path
import shlex
import time

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


def _compose_runtime_ld_library_path(ctx: CoverageContext) -> str:
    """Build the runtime library search path for Python tests."""
    paths: list[Path] = []
    candidates = [
        ctx.paths.bin_dir,
        ctx.paths.install_pkg_dir / "runtime" / "lib" / "intel64",
        ctx.paths.install_pkg_dir / "runtime" / "3rdparty" / "tbb" / "lib",
    ]
    for candidate in candidates:
        if candidate.exists():
            paths.append(candidate)

    tbb_dirs = sorted({p.parent for p in ctx.paths.install_pkg_dir.rglob("libtbb.so*") if p.is_file()})
    for tbb_dir in tbb_dirs:
        if tbb_dir not in paths:
            paths.append(tbb_dir)

    return ":".join(str(p) for p in paths)


def _expand(value: str) -> str:
    """Expand environment variables in a config value."""
    return os.path.expandvars(value)


def _selected_test_names() -> list[str]:
    """Read the optional Python test selection from the environment."""
    raw = os.environ.get("PY_TEST_NAMES", "").strip()
    if not raw:
        return []
    return [name.strip() for name in raw.split(",") if name.strip()]


def _find_openvino_wheel(wheels_dir: Path) -> Path:
    """Return the generated OpenVINO wheel from the install output."""
    candidates = sorted(wheels_dir.glob("openvino-*.whl"))
    if not candidates:
        raise FileNotFoundError(f"OpenVINO wheel not found in {wheels_dir}")
    return candidates[0]


def _write_duration_report(ctx: CoverageContext, rows: list[tuple[str, str, float]]) -> None:
    """Write per-test duration data for the current Python run."""
    report_path = ctx.workspace / "python-test-durations.csv"
    with report_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["test_name", "status", "duration_seconds", "duration_minutes"])
        for test_name, status, duration_seconds in rows:
            writer.writerow([test_name, status, f"{duration_seconds:.3f}", f"{duration_seconds / 60.0:.3f}"])

    ctx.io.export_env("PY_TEST_DURATION_REPORT", str(report_path))


def _write_stats_report(
    ctx: CoverageContext,
    *,
    total: int,
    passed: int,
    failed: int,
    skipped: int,
    not_run: int = 0,
) -> None:
    """Write aggregate Python execution counters."""
    report_path = ctx.workspace / "python-coverage-stats.env"
    report_path.write_text(
        "\n".join(
            [
                f"PY_TESTS_TOTAL={total}",
                f"PY_TESTS_PASSED={passed}",
                f"PY_TESTS_FAILED={failed}",
                f"PY_TESTS_SKIPPED={skipped}",
                f"PY_TESTS_NOT_RUN={not_run}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _run_pytest(test_name: str, target: str, args: str, env_assignments: str) -> tuple[int, float]:
    """Run one pytest-based test group and measure its duration."""
    cmd = ["python3", "-m", "pytest", "-ra", "--durations=50", target]
    if args:
        cmd.extend(shlex.split(args))
    env = env_from_assignments(env_assignments)

    print(f"========== Running Python test: {test_name} ==========")
    started_at = time.monotonic()
    rc = run_cmd(cmd, env=env, check=False)
    return rc, time.monotonic() - started_at


def _run_python_command(test_name: str, command: str, env_assignments: str) -> tuple[int, float]:
    """Run one configured shell command for the Python suite."""
    merged = command.strip()
    if env_assignments:
        merged = f"{env_assignments} {merged}"

    print(f"========== Running Python test: {test_name} ==========")
    started_at = time.monotonic()
    rc = run_cmd(["bash", "-lc", merged], check=False)
    return rc, time.monotonic() - started_at


def run(ctx: CoverageContext) -> None:
    """Execute configured Python tests and export coverage results."""
    config = ctx.workspace / "tools" / "coverage" / "config" / "tests_python.yml"
    tests = load_python_tests(config, ctx.test_profile)
    selected_names = _selected_test_names()
    if selected_names:
        selected_set = set(selected_names)
        known_names = {test.name for test in tests}
        missing_names = [name for name in selected_names if name not in known_names]
        if missing_names:
            warn("Requested Python tests were not found in config: " + ", ".join(missing_names))
        tests = [test for test in tests if test.name in selected_set]

    if not any(test.enabled for test in tests):
        skipped = [f"{test.name} ({test.skip_reason})" for test in tests]
        _write_duration_report(ctx, [(test.name, "skipped", 0.0) for test in tests])
        ctx.io.export_env("PY_TESTS_TOTAL", str(len(skipped)))
        ctx.io.export_env("PY_TESTS_PASSED", "0")
        ctx.io.export_env("PY_TESTS_FAILED", "0")
        ctx.io.export_env("PY_TESTS_SKIPPED", str(len(skipped)))
        ctx.io.export_env("PY_TESTS_NOT_RUN", "0")
        _write_stats_report(ctx, total=len(skipped), passed=0, failed=0, skipped=len(skipped), not_run=0)

        lines = [
            "",
            "## Python coverage test execution summary",
            f"Python test selection: {', '.join(selected_names) if selected_names else 'all enabled tests'}",
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

    runtime_ld_library_path = _compose_runtime_ld_library_path(ctx)
    os.environ["LD_LIBRARY_PATH"] = f"{runtime_ld_library_path}:{os.environ.get('LD_LIBRARY_PATH', '')}".rstrip(":")
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
    duration_rows: list[tuple[str, str, float]] = []

    for test in tests:
        if not test.enabled:
            skipped.append(f"{test.name} ({test.skip_reason})")
            duration_rows.append((test.name, "skipped", 0.0))
            continue

        target = _expand(test.target)
        args = _expand(test.args)
        test_env = _expand(test.env)
        command = _expand(test.command)

        if test.kind == "pytest":
            executed += 1
            rc, duration_seconds = _run_pytest(test.name, target, args, test_env)
            if rc != 0:
                failed.append(f"{test.name} (exit {rc})")
                duration_rows.append((test.name, "failed", duration_seconds))
            else:
                duration_rows.append((test.name, "passed", duration_seconds))
        elif test.kind == "pytest_if_dir":
            if not Path(target).is_dir():
                warn(f"Skipping Python test group '{test.name}' (missing: {target})")
                skipped.append(f"{test.name} (missing path)")
                duration_rows.append((test.name, "skipped", 0.0))
                continue
            executed += 1
            rc, duration_seconds = _run_pytest(test.name, target, args, test_env)
            if rc != 0:
                failed.append(f"{test.name} (exit {rc})")
                duration_rows.append((test.name, "failed", duration_seconds))
            else:
                duration_rows.append((test.name, "passed", duration_seconds))
        elif test.kind == "command":
            executed += 1
            rc, duration_seconds = _run_python_command(test.name, command, test_env)
            if rc != 0:
                failed.append(f"{test.name} (exit {rc})")
                duration_rows.append((test.name, "failed", duration_seconds))
            else:
                duration_rows.append((test.name, "passed", duration_seconds))
        else:
            warn(f"Unknown Python test kind '{test.kind}' for '{test.name}', skipping")
            skipped.append(f"{test.name} (unknown kind: {test.kind})")
            duration_rows.append((test.name, "skipped", 0.0))

    run_cmd(["python3", "-m", "coverage", "xml", "-o", str(ctx.workspace / "python-coverage.xml")])
    _write_duration_report(ctx, duration_rows)

    total_failed = len(failed)
    total_skipped = len(skipped)
    total_passed = executed - total_failed

    ctx.io.export_env("PY_TESTS_TOTAL", str(executed + total_skipped))
    ctx.io.export_env("PY_TESTS_PASSED", str(total_passed))
    ctx.io.export_env("PY_TESTS_FAILED", str(total_failed))
    ctx.io.export_env("PY_TESTS_SKIPPED", str(total_skipped))
    ctx.io.export_env("PY_TESTS_NOT_RUN", "0")
    _write_stats_report(
        ctx,
        total=executed + total_skipped,
        passed=total_passed,
        failed=total_failed,
        skipped=total_skipped,
        not_run=0,
    )

    lines = [
        "",
        "## Python coverage test execution summary",
        f"Python test selection: {', '.join(selected_names) if selected_names else 'all enabled tests'}",
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
