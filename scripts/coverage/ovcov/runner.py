# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
import os
import shlex
import subprocess
from pathlib import Path


def log(message: str) -> None:
    print(f"[coverage] {message}")


def warn(message: str) -> None:
    print(f"[coverage][warn] {message}")


def error(message: str) -> None:
    print(f"[coverage][error] {message}")


def run_cmd(
    cmd: Sequence[str] | str,
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
    shell: bool = False,
) -> int:
    display = cmd if isinstance(cmd, str) else " ".join(shlex.quote(part) for part in cmd)
    log(f"$ {display}")
    completed = subprocess.run(cmd, cwd=cwd, env=env, shell=shell, text=True)
    if check and completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {display}")
    return completed.returncode


def run_cmd_capture(cmd: Sequence[str], *, cwd: Path | None = None, check: bool = True) -> str:
    display = " ".join(shlex.quote(part) for part in cmd)
    log(f"$ {display}")
    completed = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if check and completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {display}\n{stderr}")
    return completed.stdout


def env_from_assignments(assignments: str | None, base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(base_env or os.environ)
    if not assignments:
        return env

    for token in shlex.split(assignments):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        env[key] = value
    return env
