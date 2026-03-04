# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil

from coverage_workflow import CoverageContext, env_from_assignments, load_cpp_tests, run_cmd, warn


def _remove_gcda(root: Path) -> None:
    if not root.exists():
        return
    for gcda in root.rglob("*.gcda"):
        try:
            gcda.unlink()
        except OSError:
            pass


def _run_test(
    ctx: CoverageContext,
    name: str,
    binary: str,
    mode: str,
    args: str,
    extra_env: str,
    *,
    executed: list[str],
    skipped: list[str],
    failed: list[str],
) -> None:
    exe = ctx.paths.bin_dir / binary
    if not exe.exists() or not os.access(exe, os.X_OK):
        skipped.append(f"{name} (missing binary: {binary})")
        return

    executed.append(name)
    print(f"========== Running {name} ==========")

    if mode == "raw":
        cmd = [str(exe)]
        if args:
            cmd.extend(shlex.split(args))
    else:
        cmd = [str(exe)]
        if args:
            cmd.append(f"--gtest_filter={args}")

    env = env_from_assignments(extra_env)
    rc = run_cmd(cmd, env=env, check=False)
    if rc != 0:
        failed.append(f"{name} (exit {rc})")


def run(ctx: CoverageContext) -> None:
    config = ctx.workspace / "scripts" / "coverage" / "config" / "tests_cpp.yml"
    tests = load_cpp_tests(config, ctx.test_profile)

    _remove_gcda(ctx.paths.build_dir)
    _remove_gcda(ctx.paths.build_js_dir)
    shutil.rmtree(ctx.paths.build_dir / "gcov", ignore_errors=True)

    executed: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []

    for test in tests:
        if not test.enabled:
            skipped.append(f"{test.name} ({test.skip_reason})")
            continue

        args = test.args.replace("__MODEL_PATH__", str(ctx.paths.model_path))
        _run_test(
            ctx,
            test.name,
            test.binary,
            test.mode,
            args,
            test.extra_env,
            executed=executed,
            skipped=skipped,
            failed=failed,
        )

    total_executed = len(executed)
    total_failed = len(failed)
    total_skipped = len(skipped)
    total_passed = total_executed - total_failed
    total_planned = total_executed + total_skipped

    ctx.io.export_env("CXX_TESTS_TOTAL", str(total_planned))
    ctx.io.export_env("CXX_TESTS_EXECUTED", str(total_executed))
    ctx.io.export_env("CXX_TESTS_PASSED", str(total_passed))
    ctx.io.export_env("CXX_TESTS_FAILED", str(total_failed))
    ctx.io.export_env("CXX_TESTS_SKIPPED", str(total_skipped))

    lines = [
        "## C++ coverage test execution summary",
        "",
        f"Test profile: {ctx.test_profile}",
        "",
        f"GPU mode: {'true' if ctx.run_gpu_tests else 'false'}",
        "",
        f"NPU mode: {'true' if ctx.run_npu_tests else 'false'}",
        "",
        f"C++ tests planned: {total_planned}",
        f"C++ tests executed: {total_executed}",
        f"C++ tests passed: {total_passed}",
        f"C++ tests failed: {total_failed}",
        f"C++ tests skipped: {total_skipped}",
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
        warn("One or more C++ tests failed; continuing to coverage generation.")
