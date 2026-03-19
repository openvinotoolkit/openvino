# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from coverage_workflow import CoverageContext, run_cmd


def _find_openvino_wheel(wheels_dir: Path) -> Path:
    """Return the generated OpenVINO wheel from the install output."""
    candidates = sorted(wheels_dir.glob("openvino-*.whl"))
    if not candidates:
        raise FileNotFoundError(f"OpenVINO wheel not found in {wheels_dir}")
    return candidates[0]


def run(ctx: CoverageContext) -> None:
    """Build, install, and package the Python and JS coverage artifacts."""
    # Build the main OpenVINO tree with coverage enabled.
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

    # Export the built Python wheel into install_pkg/wheels so shard jobs can restore and reuse it.
    # This only produces the wheel artifact; it does not install the package into the current Python environment.
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

    # Install the main runtime and development files into the reusable package directory.
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

    # Install the test binaries and assets consumed by the coverage shards.
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

    # Configure the separate JS bindings build used by Node.js coverage tests.
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
            "-DENABLE_INTEL_GPU=ON" if ctx.run_gpu_tests else "-DENABLE_INTEL_GPU=OFF",
            js_npu_flag,
            "-DENABLE_JS=ON",
            "-DENABLE_COVERAGE=ON",
            f"-DCMAKE_INSTALL_PREFIX={ctx.paths.js_dir / 'bin'}",
        ]
    )

    # Build the JS bindings package.
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

    # Install the JS bindings into the location restored by JS coverage shards.
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

    # Install the freshly built wheel into the current runner environment so subsequent Python-based
    # steps import this build of OpenVINO instead of any preinstalled version.
    run_cmd(["python3", "-m", "pip", "install", "--force-reinstall", str(wheel)])

    # Print the main binary directory contents for quick diagnostics in CI logs.
    run_cmd(["ls", "-la", str(ctx.paths.bin_dir)])
