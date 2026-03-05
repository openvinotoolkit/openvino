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


def _compose_runtime_ld_library_path(ctx: CoverageContext) -> str:
    paths: list[Path] = []

    candidates = [
        ctx.paths.bin_dir,
        ctx.paths.install_pkg_dir / "runtime" / "lib" / "intel64",
        ctx.paths.install_pkg_dir / "runtime" / "3rdparty" / "tbb" / "lib",
    ]
    for candidate in candidates:
        if candidate.exists():
            paths.append(candidate)

    # Find packaged TBB locations across layout variants.
    tbb_dirs = sorted({p.parent for p in ctx.paths.install_pkg_dir.rglob("libtbb.so*") if p.is_file()})
    for tbb_dir in tbb_dirs:
        if tbb_dir not in paths:
            paths.append(tbb_dir)

    return ":".join(str(p) for p in paths)


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
    runtime_ld_library_path: str,
) -> None:
    exe = ctx.paths.bin_dir / binary
    if not exe.exists():
        skipped.append(f"{name} (missing binary: {binary})")
        return

    if not os.access(exe, os.X_OK):
        # GitHub artifact download may drop execute bits; try to recover in-place.
        try:
            mode = exe.stat().st_mode
            exe.chmod(mode | 0o111)
        except OSError:
            pass

    if not os.access(exe, os.X_OK):
        skipped.append(f"{name} (binary not executable: {binary})")
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
    existing_ld = env.get("LD_LIBRARY_PATH", "")
    if runtime_ld_library_path:
        env["LD_LIBRARY_PATH"] = f"{runtime_ld_library_path}:{existing_ld}" if existing_ld else runtime_ld_library_path

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
    runtime_ld_library_path = _compose_runtime_ld_library_path(ctx)

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
            runtime_ld_library_path=runtime_ld_library_path,
        )

    total_executed = len(executed)
    total_failed = len(failed)
    total_skipped = len(skipped)
    total_passed = total_executed - total_failed
    total_planned = total_executed + total_skipped

    if total_planned > 0 and total_executed == 0:
        warn(
            f"No C++ tests were executed (all skipped). "
            f"Check restored binaries under: {ctx.paths.bin_dir}"
        )

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
