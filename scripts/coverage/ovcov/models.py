# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SUPPORTED_PROFILES = {"cpu", "cpu_gpu", "cpu_npu", "cpu_npu_gpu"}


@dataclass(frozen=True)
class CppTestCase:
    name: str
    enabled: bool
    skip_reason: str
    binary: str
    mode: str
    args: str
    extra_env: str


@dataclass(frozen=True)
class PythonTestCase:
    name: str
    enabled: bool
    skip_reason: str
    kind: str
    target: str
    args: str
    env: str
    command: str


@dataclass(frozen=True)
class JsTestCase:
    name: str
    enabled: bool
    skip_reason: str
    kind: str
    command: str


@dataclass
class StepResult:
    success: bool
    message: str = ""


@dataclass
class TestStats:
    total_executed: int
    failed: list[str]
    skipped: list[str]

    @property
    def passed(self) -> int:
        return self.total_executed - len(self.failed)

    @property
    def total_planned(self) -> int:
        return self.total_executed + len(self.skipped)


@dataclass(frozen=True)
class Paths:
    workspace: Path
    build_dir: Path
    build_js_dir: Path
    install_pkg_dir: Path
    bin_dir: Path
    js_dir: Path
    model_path: Path


@dataclass(frozen=True)
class ProfileFlags:
    run_gpu_tests: bool
    run_npu_tests: bool
    gpu_flags: tuple[str, ...]
    npu_flags: tuple[str, ...]
