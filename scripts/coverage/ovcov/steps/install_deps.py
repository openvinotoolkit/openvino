# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

from ..context import CoverageContext
from ..runner import run_cmd


def run(ctx: CoverageContext) -> None:
    sudo_prefix: list[str] = []
    if os.geteuid() != 0:
        sudo_prefix = ["sudo"]

    run_cmd([*sudo_prefix, "apt", "--assume-yes", "update"])
    if sudo_prefix:
        run_cmd([*sudo_prefix, "-E", str(ctx.workspace / "install_build_dependencies.sh")])
    else:
        run_cmd([str(ctx.workspace / "install_build_dependencies.sh")])
    run_cmd(
        [
            *sudo_prefix,
            "apt",
            "--assume-yes",
            "install",
            "lcov",
            "wget",
            "pigz",
            "xvfb",
            "clang-14",
            "libclang-14-dev",
            "clinfo",
            "ca-certificates",
        ]
    )

    run_cmd(["python3", "-m", "pip", "install", "--upgrade", "pip"])
    run_cmd(["python3", "-m", "pip", "install", "pyyaml", "pytest", "pytest-cov", "pytest-xdist[psutil]"])
    run_cmd(["python3", "-m", "pip", "install", "-r", str(ctx.workspace / "src/bindings/python/wheel/requirements-dev.txt")])
    run_cmd(["python3", "-m", "pip", "install", "-r", str(ctx.workspace / "src/frontends/paddle/tests/requirements.txt")])
    run_cmd(["python3", "-m", "pip", "install", "-r", str(ctx.workspace / "src/frontends/onnx/tests/requirements.txt")])
    run_cmd(["python3", "-m", "pip", "install", "-r", str(ctx.workspace / "src/frontends/tensorflow/tests/requirements.txt")])
    run_cmd(["python3", "-m", "pip", "install", "-r", str(ctx.workspace / "src/frontends/tensorflow_lite/tests/requirements.txt")])
    run_cmd(["python3", "-m", "pip", "install", "-r", str(ctx.workspace / "tests/requirements_jax")])
