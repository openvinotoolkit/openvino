# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tarfile

from coverage_workflow import CoverageContext


def run(ctx: CoverageContext) -> None:
    """Package generated coverage reports into a single archive."""
    report_dir = ctx.workspace / "coverage-report"
    coverage_info = ctx.workspace / "coverage.info"
    archive = ctx.workspace / "coverage-report.tgz"

    if report_dir.exists() and coverage_info.exists():
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(report_dir, arcname="coverage-report")
            tar.add(coverage_info, arcname="coverage.info")
            py_cov = ctx.workspace / "python-coverage.xml"
            js_cov = ctx.workspace / "js-lcov.info"
            if py_cov.exists():
                tar.add(py_cov, arcname="python-coverage.xml")
            if js_cov.exists():
                tar.add(js_cov, arcname="js-lcov.info")

        ctx.io.append_summary(
            "\n"
            "Artifact: `coverage-report.tgz` (download from Actions -> Artifacts, then extract and open `coverage-report/index.html`).\n"
        )
    else:
        ctx.io.append_summary(
            "\n"
            "coverage-report/ or coverage.info missing -> skipping artifact packaging.\n"
        )
