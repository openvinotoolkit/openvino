# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from ..context import CoverageContext
from ..runner import run_cmd, warn


def _has_gcda(root: Path) -> bool:
    if not root.exists():
        return False
    return any(root.rglob("*.gcda"))


def run(ctx: CoverageContext) -> None:
    src_dir = ctx.workspace
    report_dir = ctx.workspace / "coverage-report"
    main_info = ctx.workspace / "coverage-cpp-main.info"
    js_info = ctx.workspace / "coverage-cpp-js.info"
    merged_info = ctx.workspace / "coverage.info"

    run_cmd(
        [
            "lcov",
            "--capture",
            "--directory",
            str(ctx.paths.build_dir),
            "--build-directory",
            str(ctx.paths.build_dir),
            "--base-directory",
            str(src_dir),
            "--output-file",
            str(main_info),
            "--no-external",
            "--rc",
            "geninfo_unexecuted_blocks=1",
            "--ignore-errors",
            "mismatch,negative,unused,gcov",
        ]
    )

    if _has_gcda(ctx.paths.build_js_dir):
        run_cmd(
            [
                "lcov",
                "--capture",
                "--directory",
                str(ctx.paths.build_js_dir),
                "--build-directory",
                str(ctx.paths.build_js_dir),
                "--base-directory",
                str(src_dir),
                "--output-file",
                str(js_info),
                "--no-external",
                "--rc",
                "geninfo_unexecuted_blocks=1",
                "--ignore-errors",
                "mismatch,negative,unused,gcov",
            ]
        )
        run_cmd(["lcov", "-a", str(main_info), "-a", str(js_info), "-o", str(merged_info)])
    else:
        warn(f"No .gcda files found in {ctx.paths.build_js_dir}, skipping JS native C++ capture")
        merged_info.write_bytes(main_info.read_bytes())

    run_cmd(
        [
            "lcov",
            "--remove",
            str(merged_info),
            "--ignore-errors",
            "unused,mismatch",
            f"{src_dir}/build/*",
            f"{src_dir}/build_js/*",
            f"{src_dir}/*.pb.cc",
            f"{src_dir}/*.pb.h",
            f"{src_dir}/*/tests/*",
            f"{src_dir}/tests/*",
            f"{src_dir}/docs/*",
            f"{src_dir}/samples/*",
            f"{src_dir}/tools/*",
            f"{src_dir}/src/bindings/js/node/tests/*",
            f"{src_dir}/src/bindings/python/tests/*",
            f"{src_dir}/thirdparty/*",
            "-o",
            str(merged_info),
        ]
    )

    if main_info.exists():
        main_info.unlink()
    if js_info.exists():
        js_info.unlink()

    run_cmd(["grep", "-m", "5", f"^SF:{src_dir}/build/", str(merged_info)], check=False)
    run_cmd(["grep", "-m", "5", f"^SF:{src_dir}/build_js/", str(merged_info)], check=False)

    report_dir.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            "genhtml",
            str(merged_info),
            "--output-directory",
            str(report_dir),
            "--prefix",
            str(src_dir),
            "--synthesize-missing",
        ]
    )
