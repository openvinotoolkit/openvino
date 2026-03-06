# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

from coverage_workflow import CoverageContext, run_cmd_capture


def _int_env(name: str) -> int:
    raw = os.environ.get(name, "0").strip()
    if not raw:
        return 0
    try:
        return int(raw)
    except ValueError:
        return 0


def run(ctx: CoverageContext) -> None:
    cxx_total = _int_env("CXX_TESTS_TOTAL")
    cxx_passed = _int_env("CXX_TESTS_PASSED")
    cxx_failed = _int_env("CXX_TESTS_FAILED")
    cxx_skipped = _int_env("CXX_TESTS_SKIPPED")

    py_total = _int_env("PY_TESTS_TOTAL")
    py_passed = _int_env("PY_TESTS_PASSED")
    py_failed = _int_env("PY_TESTS_FAILED")
    py_skipped = _int_env("PY_TESTS_SKIPPED")

    js_total = _int_env("JS_TESTS_TOTAL")
    js_passed = _int_env("JS_TESTS_PASSED")
    js_failed = _int_env("JS_TESTS_FAILED")
    js_skipped = _int_env("JS_TESTS_SKIPPED")

    overall_total = cxx_total + py_total + js_total
    overall_passed = cxx_passed + py_passed + js_passed
    overall_failed = cxx_failed + py_failed + js_failed
    overall_skipped = cxx_skipped + py_skipped + js_skipped

    if overall_total > 0:
        overall_rate = f"{(overall_passed * 100.0) / overall_total:.1f}"
    else:
        overall_rate = "0.0"

    lines = [
        "## Coverage Report Summary",
        "",
        f"**Profile:** `{ctx.test_profile}`  ",
        f"**Overall pass rate:** `{overall_rate}%`",
        "",
        "### Overall",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Total test units | {overall_total} |",
        f"| Passed | {overall_passed} |",
        f"| Failed | {overall_failed} |",
        f"| Skipped | {overall_skipped} |",
        "",
        "### By Suite",
        "| Suite | Total | Passed | Failed | Skipped |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| C++ | {cxx_total} | {cxx_passed} | {cxx_failed} | {cxx_skipped} |",
        f"| Python | {py_total} | {py_passed} | {py_failed} | {py_skipped} |",
        f"| JS | {js_total} | {js_passed} | {js_failed} | {js_skipped} |",
        "",
        "### Coverage Files",
        "- C/C++ lcov: `coverage.info`",
        "- Python XML: `python-coverage.xml`",
        "- JS lcov: `js-lcov.info`",
        "",
    ]

    if not (ctx.workspace / "python-coverage.xml").exists():
        lines.append(":warning: Python coverage file is missing (`python-coverage.xml`).")
    if not (ctx.workspace / "js-lcov.info").exists():
        lines.append(":warning: JS coverage file is missing (`js-lcov.info`).")

    lines.extend(["", "### C/C++ Coverage Details (lcov)"])

    coverage_info = ctx.workspace / "coverage.info"
    if coverage_info.exists():
        summary_output = run_cmd_capture(["lcov", "--summary", str(coverage_info)], check=False)
        lines.append("```")
        lines.append(summary_output.rstrip())
        lines.append("```")
    else:
        lines.append("coverage.info not found (coverage generation likely failed earlier).")

    ctx.io.append_summary("\n".join(lines) + "\n")
