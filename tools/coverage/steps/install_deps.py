# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import shlex
import shutil
import subprocess

from coverage_workflow import CoverageContext, run_cmd, warn


def _flag_from_env(name: str, default: str = "false") -> bool:
    value = os.environ.get(name, default).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _node_major_version() -> int | None:
    node_path = shutil.which("node")
    if not node_path:
        return None

    completed = subprocess.run([node_path, "--version"], text=True, capture_output=True)
    if completed.returncode != 0:
        return None

    raw = completed.stdout.strip()
    if raw.startswith("v"):
        raw = raw[1:]
    major = raw.split(".", 1)[0]
    if not major.isdigit():
        return None
    return int(major)


def _install_nodejs(sudo_prefix: list[str], major_version: str) -> None:
    if not major_version.isdigit():
        raise RuntimeError(f"Invalid OVCOV_NODEJS_VERSION='{major_version}'. Expected a major number, for example: 22")

    run_cmd([*sudo_prefix, "apt", "--assume-yes", "install", "curl", "gnupg"])

    setup_script = f"https://deb.nodesource.com/setup_{major_version}.x"
    if sudo_prefix:
        shell_cmd = f"curl -fsSL {shlex.quote(setup_script)} | sudo -E bash -"
    else:
        shell_cmd = f"curl -fsSL {shlex.quote(setup_script)} | bash -"
    run_cmd(shell_cmd, shell=True)

    run_cmd([*sudo_prefix, "apt", "--assume-yes", "install", "nodejs"])
    run_cmd(["node", "--version"])
    run_cmd(["npm", "--version"])


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

    install_nodejs = _flag_from_env("OVCOV_INSTALL_NODEJS")
    nodejs_version = os.environ.get("OVCOV_NODEJS_VERSION", "22").strip() or "22"

    if install_nodejs:
        current_major = _node_major_version()
        desired_major = int(nodejs_version) if nodejs_version.isdigit() else None
        if current_major is not None and desired_major is not None and current_major >= desired_major:
            run_cmd(["node", "--version"])
            run_cmd(["npm", "--version"])
        else:
            _install_nodejs(sudo_prefix, nodejs_version)
    elif shutil.which("node") is None or shutil.which("npm") is None:
        warn(
            "Node.js/npm are not installed. JS coverage tests will fail. "
            "Use '--install-nodejs --nodejs-version 22' when running install-deps locally."
        )
