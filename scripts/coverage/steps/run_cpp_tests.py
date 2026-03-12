# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import csv
import os
from pathlib import Path
import re
import shlex
import shutil
import time

from coverage_workflow import CoverageContext, CppTestCase, env_from_assignments, load_cpp_tests, run_cmd, warn


@dataclass(frozen=True)
class _CppTestRunResult:
    name: str
    status: str = "passed"
    skipped: str = ""
    failed: str = ""
    duration_seconds: float = 0.0


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


def _slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "test"


def _gcov_stage_root(ctx: CoverageContext) -> Path:
    return ctx.workspace / ".tmp" / "cpp-gcov"


def _selected_test_names() -> list[str]:
    raw = os.environ.get("CXX_TEST_NAMES", "").strip()
    if not raw:
        return []
    return [name.strip() for name in raw.split(",") if name.strip()]


def _gcov_prefix_strip(path: Path) -> int:
    return len([part for part in path.resolve().parts if part != os.sep])


def _build_gcov_env(env: dict[str, str], *, gcov_dir: Path | None, workspace: Path) -> dict[str, str]:
    prepared = dict(env)
    if gcov_dir is None:
        prepared.pop("GCOV_PREFIX", None)
        prepared.pop("GCOV_PREFIX_STRIP", None)
        return prepared

    gcov_dir.mkdir(parents=True, exist_ok=True)
    prepared["GCOV_PREFIX"] = str(gcov_dir)
    prepared["GCOV_PREFIX_STRIP"] = str(_gcov_prefix_strip(workspace))
    return prepared


def _build_test_command(exe: Path, *, mode: str, args: str) -> list[str]:
    cmd = [str(exe)]
    if mode == "raw":
        if args:
            cmd.extend(shlex.split(args))
        return cmd

    if args:
        cmd.append(f"--gtest_filter={args}")
    return cmd


def _run_test(
    ctx: CoverageContext,
    test: CppTestCase,
    args: str,
    *,
    runtime_ld_library_path: str,
    gcov_dir: Path | None,
) -> _CppTestRunResult:
    exe = ctx.paths.bin_dir / test.binary
    if not exe.exists():
        return _CppTestRunResult(
            name=test.name,
            status="skipped",
            skipped=f"{test.name} (missing binary: {test.binary})",
        )

    if not os.access(exe, os.X_OK):
        # GitHub artifact download may drop execute bits; try to recover in-place.
        try:
            file_mode = exe.stat().st_mode
            exe.chmod(file_mode | 0o111)
        except OSError:
            pass

    if not os.access(exe, os.X_OK):
        return _CppTestRunResult(
            name=test.name,
            status="skipped",
            skipped=f"{test.name} (binary not executable: {test.binary})",
        )

    print(f"========== Running {test.name} ==========")
    cmd = _build_test_command(exe, mode=test.mode, args=args)
    env = env_from_assignments(test.extra_env)
    existing_ld = env.get("LD_LIBRARY_PATH", "")
    if runtime_ld_library_path:
        env["LD_LIBRARY_PATH"] = f"{runtime_ld_library_path}:{existing_ld}" if existing_ld else runtime_ld_library_path
    env = _build_gcov_env(env, gcov_dir=gcov_dir, workspace=ctx.workspace)

    started_at = time.monotonic()
    rc = run_cmd(cmd, env=env, check=False)
    duration_seconds = time.monotonic() - started_at
    if rc != 0:
        return _CppTestRunResult(
            name=test.name,
            status="failed",
            failed=f"{test.name} (exit {rc})",
            duration_seconds=duration_seconds,
        )
    return _CppTestRunResult(name=test.name, status="passed", duration_seconds=duration_seconds)


def _write_duration_report(ctx: CoverageContext, results: list[_CppTestRunResult]) -> None:
    report_path = ctx.workspace / "cpp-test-durations.csv"
    with report_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["test_name", "status", "duration_seconds", "duration_minutes"])
        for result in results:
            writer.writerow([result.name, result.status, f"{result.duration_seconds:.3f}", f"{result.duration_seconds / 60.0:.3f}"])

    total_duration = sum(result.duration_seconds for result in results)
    ctx.io.export_env("CXX_TEST_DURATION_REPORT", str(report_path))
    ctx.io.export_env("CXX_TEST_DURATION_TOTAL_SECONDS", f"{total_duration:.3f}")


def _write_stats_report(
    ctx: CoverageContext,
    *,
    total: int,
    executed: int,
    passed: int,
    failed: int,
    skipped: int,
    not_run: int = 0,
) -> None:
    report_path = ctx.workspace / "cpp-coverage-stats.env"
    report_path.write_text(
        "\n".join(
            [
                f"CXX_TESTS_TOTAL={total}",
                f"CXX_TESTS_EXECUTED={executed}",
                f"CXX_TESTS_PASSED={passed}",
                f"CXX_TESTS_FAILED={failed}",
                f"CXX_TESTS_SKIPPED={skipped}",
                f"CXX_TESTS_NOT_RUN={not_run}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _run_tests_serial(
    ctx: CoverageContext,
    tests: list[tuple[CppTestCase, str]],
    *,
    runtime_ld_library_path: str,
) -> tuple[list[str], list[str], list[str]]:
    results: list[_CppTestRunResult] = []
    executed: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []

    for test, args in tests:
        result = _run_test(
            ctx,
            test,
            args,
            runtime_ld_library_path=runtime_ld_library_path,
            gcov_dir=None,
        )
        if result.skipped:
            skipped.append(result.skipped)
            results.append(result)
            continue
        executed.append(result.name)
        if result.failed:
            failed.append(result.failed)
        results.append(result)

    _write_duration_report(ctx, results)
    return executed, skipped, failed


def _run_tests_parallel(
    ctx: CoverageContext,
    tests: list[tuple[CppTestCase, str]],
    *,
    runtime_ld_library_path: str,
) -> tuple[list[str], list[str], list[str]]:
    results: list[_CppTestRunResult] = []
    executed: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []

    stage_root = _gcov_stage_root(ctx)
    runs_root = stage_root / "runs"
    shutil.rmtree(stage_root, ignore_errors=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    workers = min(ctx.cpp_test_concurrency, len(tests))
    print(
        f"[coverage] Running {len(tests)} C++ test binaries with concurrency={workers} "
        f"using staged gcov output under {runs_root}"
    )

    futures = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, (test, args) in enumerate(tests, start=1):
            gcov_dir = runs_root / f"{index:03d}-{_slugify(test.name)}"
            future = executor.submit(
                _run_test,
                ctx,
                test,
                args,
                runtime_ld_library_path=runtime_ld_library_path,
                gcov_dir=gcov_dir,
            )
            futures[future] = (index, test.name)

        ordered_results: dict[int, _CppTestRunResult] = {}
        for future in as_completed(futures):
            index, name = futures[future]
            try:
                ordered_results[index] = future.result()
            except Exception as exc:  # noqa: BLE001
                ordered_results[index] = _CppTestRunResult(name=name, failed=f"{name} (runner error: {exc})")

    for index in sorted(ordered_results):
        result = ordered_results[index]
        if result.skipped:
            skipped.append(result.skipped)
            results.append(result)
            continue
        executed.append(result.name)
        if result.failed:
            failed.append(result.failed)
        results.append(result)

    _write_duration_report(ctx, results)
    return executed, skipped, failed


def run(ctx: CoverageContext) -> None:
    config = ctx.workspace / "scripts" / "coverage" / "config" / "tests_cpp.yml"
    tests = load_cpp_tests(config, ctx.test_profile)
    selected_names = _selected_test_names()
    if selected_names:
        selected_set = set(selected_names)
        known_names = {test.name for test in tests}
        missing_names = [name for name in selected_names if name not in known_names]
        if missing_names:
            warn("Requested C++ tests were not found in config: " + ", ".join(missing_names))
        tests = [test for test in tests if test.name in selected_set]

    _remove_gcda(ctx.paths.build_dir)
    _remove_gcda(ctx.paths.build_js_dir)
    shutil.rmtree(ctx.paths.build_dir / "gcov", ignore_errors=True)
    shutil.rmtree(_gcov_stage_root(ctx), ignore_errors=True)

    runtime_ld_library_path = _compose_runtime_ld_library_path(ctx)
    enabled_tests: list[tuple[CppTestCase, str]] = []
    skipped: list[str] = []

    for test in tests:
        if not test.enabled:
            skipped.append(f"{test.name} ({test.skip_reason})")
            continue

        args = test.args.replace("__MODEL_PATH__", str(ctx.paths.model_path))
        enabled_tests.append((test, args))

    if ctx.cpp_test_concurrency > 1 and enabled_tests:
        executed, dynamic_skips, failed = _run_tests_parallel(
            ctx,
            enabled_tests,
            runtime_ld_library_path=runtime_ld_library_path,
        )
    else:
        executed, dynamic_skips, failed = _run_tests_serial(
            ctx,
            enabled_tests,
            runtime_ld_library_path=runtime_ld_library_path,
        )

    skipped.extend(dynamic_skips)

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
    ctx.io.export_env("CXX_TESTS_NOT_RUN", "0")
    _write_stats_report(
        ctx,
        total=total_planned,
        executed=total_executed,
        passed=total_passed,
        failed=total_failed,
        skipped=total_skipped,
        not_run=0,
    )

    lines = [
        "## C++ coverage test execution summary",
        "",
        f"Test profile: {ctx.test_profile}",
        "",
        f"GPU mode: {'true' if ctx.run_gpu_tests else 'false'}",
        "",
        f"NPU mode: {'true' if ctx.run_npu_tests else 'false'}",
        "",
        f"C++ test concurrency: {max(1, ctx.cpp_test_concurrency)}",
        "",
        f"C++ test selection: {', '.join(selected_names) if selected_names else 'all enabled tests'}",
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
