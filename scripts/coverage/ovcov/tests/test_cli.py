# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from ovcov.cli import build_parser


def test_run_all_parser_accepts_range_and_profile() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "run-all",
            "--profile",
            "cpu_npu_gpu",
            "--install-deps",
            "--from",
            "configure",
            "--to",
            "write-summary",
        ]
    )
    assert args.command == "run-all"
    assert args.profile == "cpu_npu_gpu"
    assert args.install_deps is True
    assert args.from_step == "configure"
    assert args.to_step == "write-summary"


def test_step_parser_accepts_known_step() -> None:
    parser = build_parser()
    args = parser.parse_args(["step", "run-cpp-tests"])
    assert args.command == "step"
    assert args.step_name == "run-cpp-tests"
