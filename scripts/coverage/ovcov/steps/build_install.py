# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from ..context import CoverageContext
from ..runner import run_cmd


def _find_openvino_wheel(wheels_dir: Path) -> Path:
    candidates = sorted(wheels_dir.glob("openvino-*.whl"))
    if not candidates:
        raise FileNotFoundError(f"OpenVINO wheel not found in {wheels_dir}")
    return candidates[0]


def run(ctx: CoverageContext) -> None:
    run_cmd(
        [
            "cmake",
            "--build",
            str(ctx.paths.build_dir),
            "--parallel",
            str(ctx.parallel_jobs),
            "--config",
            ctx.build_type,
        ]
    )

    run_cmd(
        [
            "cmake",
            "--install",
            str(ctx.paths.build_dir),
            "--prefix",
            str(ctx.paths.install_pkg_dir),
            "--component",
            "python_wheels",
            "--config",
            ctx.build_type,
        ]
    )

    run_cmd(
        [
            "cmake",
            "--install",
            str(ctx.paths.build_dir),
            "--prefix",
            str(ctx.paths.install_pkg_dir),
            "--config",
            ctx.build_type,
        ]
    )
    run_cmd(
        [
            "cmake",
            "--install",
            str(ctx.paths.build_dir),
            "--prefix",
            str(ctx.paths.install_pkg_dir),
            "--component",
            "tests",
            "--config",
            ctx.build_type,
        ]
    )

    js_npu_flag = "-DENABLE_INTEL_NPU=ON" if ctx.run_npu_tests else "-DENABLE_INTEL_NPU=OFF"

    run_cmd(
        [
            "cmake",
            "-S",
            str(ctx.workspace),
            "-B",
            str(ctx.paths.build_js_dir),
            "-GNinja",
            f"-DCMAKE_BUILD_TYPE={ctx.build_type}",
            "-DCPACK_GENERATOR=NPM",
            "-DENABLE_SYSTEM_TBB=OFF",
            "-DENABLE_TESTS=OFF",
            "-DENABLE_SAMPLES=OFF",
            "-DENABLE_WHEEL=OFF",
            "-DENABLE_PYTHON=OFF",
            "-DENABLE_INTEL_GPU=OFF",
            js_npu_flag,
            "-DENABLE_JS=ON",
            "-DENABLE_COVERAGE=ON",
            f"-DCMAKE_INSTALL_PREFIX={ctx.paths.js_dir / 'bin'}",
        ]
    )
    run_cmd(
        [
            "cmake",
            "--build",
            str(ctx.paths.build_js_dir),
            "--parallel",
            str(ctx.parallel_jobs),
            "--config",
            ctx.build_type,
        ]
    )
    run_cmd(
        [
            "cmake",
            "--install",
            str(ctx.paths.build_js_dir),
            "--prefix",
            str(ctx.paths.js_dir / 'bin'),
            "--config",
            ctx.build_type,
        ]
    )

    wheel = _find_openvino_wheel(ctx.paths.install_pkg_dir / "wheels")
    run_cmd(["python3", "-m", "pip", "install", "--force-reinstall", str(wheel)])

    run_cmd(["ls", "-la", str(ctx.paths.bin_dir)])
