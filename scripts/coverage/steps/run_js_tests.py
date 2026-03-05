# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path
import shutil

from coverage_workflow import CoverageContext, load_js_tests, run_cmd, warn


def _compose_runtime_ld_library_path(ctx: CoverageContext) -> str:
    paths: list[Path] = []
    candidates = [
        ctx.paths.bin_dir,
        ctx.paths.install_pkg_dir / "runtime" / "lib" / "intel64",
        ctx.paths.install_pkg_dir / "runtime" / "3rdparty" / "tbb" / "lib",
        ctx.paths.js_dir / "bin",
    ]
    for candidate in candidates:
        if candidate.exists():
            paths.append(candidate)

    tbb_dirs = sorted({p.parent for p in ctx.paths.install_pkg_dir.rglob("libtbb.so*") if p.is_file()})
    for tbb_dir in tbb_dirs:
        if tbb_dir not in paths:
            paths.append(tbb_dir)

    return ":".join(str(p) for p in paths)


def run(ctx: CoverageContext) -> None:
    if shutil.which("node") is None or shutil.which("npm") is None:
        raise RuntimeError(
            "Node.js/npm are not available. Install them first, for example via: "
            "`python3 scripts/coverage/coverage.py step install-deps --install-nodejs --nodejs-version 22`"
        )

    config = ctx.workspace / "scripts" / "coverage" / "config" / "tests_js.yml"
    tests = load_js_tests(config, ctx.test_profile)

    if not any(test.enabled for test in tests):
        skipped = [f"{test.name} ({test.skip_reason})" for test in tests]
        ctx.io.export_env("JS_TESTS_TOTAL", str(len(skipped)))
        ctx.io.export_env("JS_TESTS_PASSED", "0")
        ctx.io.export_env("JS_TESTS_FAILED", "0")
        ctx.io.export_env("JS_TESTS_SKIPPED", str(len(skipped)))

        lines = [
            "",
            "## JS coverage test execution summary",
            "JS tests executed: 0",
            "JS tests passed: 0",
            "JS tests failed: 0",
            f"JS tests skipped: {len(skipped)}",
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
        warn(f"No JS tests are enabled for TEST_PROFILE={ctx.test_profile}; skipping JS suite.")
        return

    os.environ["JS_TEST_CONCURRENCY"] = str(ctx.js_test_concurrency)
    os.environ["OV_WORKSPACE"] = str(ctx.workspace)
    runtime_ld_library_path = _compose_runtime_ld_library_path(ctx)
    os.environ["LD_LIBRARY_PATH"] = f"{runtime_ld_library_path}:{os.environ.get('LD_LIBRARY_PATH', '')}".rstrip(":")

    run_cmd(["npm", "i"], cwd=ctx.paths.js_dir)
    run_cmd(["npm", "i", "--no-save", "c8"], cwd=ctx.paths.js_dir)

    executed = 0
    skipped_count = 0
    failed: list[str] = []
    skipped: list[str] = []

    for test in tests:
        if not test.enabled:
            skipped_count += 1
            skipped.append(f"{test.name} ({test.skip_reason})")
            continue

        if test.kind != "command":
            skipped_count += 1
            skipped.append(f"{test.name} (unknown kind: {test.kind})")
            continue

        executed += 1
        print(f"========== Running JS test: {test.name} ==========")
        command = os.path.expandvars(test.command)
        rc = run_cmd(["bash", "-lc", command], cwd=ctx.paths.js_dir, check=False)
        if rc != 0:
            failed.append(f"{test.name} (exit {rc})")

    source = ctx.workspace / "js-coverage" / "lcov.info"
    target = ctx.workspace / "js-lcov.info"
    if source.exists():
        shutil.copyfile(source, target)

    total_failed = len(failed)
    total_passed = executed - total_failed

    ctx.io.export_env("JS_TESTS_TOTAL", str(executed + skipped_count))
    ctx.io.export_env("JS_TESTS_PASSED", str(total_passed))
    ctx.io.export_env("JS_TESTS_FAILED", str(total_failed))
    ctx.io.export_env("JS_TESTS_SKIPPED", str(skipped_count))

    lines = [
        "",
        "## JS coverage test execution summary",
        f"JS tests executed: {executed}",
        f"JS tests passed: {total_passed}",
        f"JS tests failed: {total_failed}",
        f"JS tests skipped: {skipped_count}",
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
        warn("One or more JS tests failed; continuing to coverage generation.")
