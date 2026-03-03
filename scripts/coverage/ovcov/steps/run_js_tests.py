# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import shutil

from ..config import load_js_tests
from ..context import CoverageContext
from ..runner import run_cmd, warn


def run(ctx: CoverageContext) -> None:
    config = ctx.workspace / "scripts" / "coverage" / "config" / "tests_js.yml"

    os.environ["JS_TEST_CONCURRENCY"] = str(ctx.js_test_concurrency)
    os.environ["OV_WORKSPACE"] = str(ctx.workspace)

    run_cmd(["npm", "i"], cwd=ctx.paths.js_dir)
    run_cmd(["npm", "i", "--no-save", "c8"], cwd=ctx.paths.js_dir)

    tests = load_js_tests(config, ctx.test_profile)

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
