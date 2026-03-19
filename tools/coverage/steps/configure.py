# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from coverage_workflow import CoverageContext, run_cmd


def run(ctx: CoverageContext) -> None:
    """Configure the main OpenVINO coverage build with CMake."""
    ctx.log_profile()

    cmd = [
        "cmake",
        "-S",
        str(ctx.workspace),
        "-B",
        str(ctx.paths.build_dir),
        "-GNinja",
        f"-DCMAKE_BUILD_TYPE={ctx.build_type}",
        "-DCMAKE_VERBOSE_MAKEFILE=ON",
        "-DENABLE_PYTHON=ON",
        "-DENABLE_JS=ON",
        "-DENABLE_TESTS=ON",
        "-DENABLE_FUNCTIONAL_TESTS=ON",
        "-DENABLE_OV_ONNX_FRONTEND=ON",
        "-DENABLE_OV_PADDLE_FRONTEND=ON",
        "-DENABLE_OV_TF_FRONTEND=ON",
        "-DENABLE_OV_TF_LITE_FRONTEND=ON",
        "-DENABLE_STRICT_DEPENDENCIES=OFF",
        "-DENABLE_COVERAGE=ON",
        f"-DCMAKE_C_COMPILER={ctx.cc}",
        f"-DCMAKE_CXX_COMPILER={ctx.cxx}",
        "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
        "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
        "-DCMAKE_C_LINKER_LAUNCHER=ccache",
        "-DCMAKE_CXX_LINKER_LAUNCHER=ccache",
        "-DENABLE_SYSTEM_SNAPPY=ON",
        *ctx.gpu_flags,
        *ctx.npu_flags,
    ]
    run_cmd(cmd)
