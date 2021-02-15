#!/usr/bin/env python3
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

""" Test inference with conditional compiled binaries.
"""
import sys
from proc_utils import cmd_exec  # pylint: disable=import-error
from install_pkg import get_openvino_environment  # pylint: disable=import-error


def test_infer(test_id, model, artifacts):
    """ Test inference with conditional compiled binaries
    """
    install_prefix = artifacts / test_id / "install_pkg"
    exe_suffix = ".exe" if sys.platform == "win32" else ""
    benchmark_app = install_prefix / "bin" / f"benchmark_app{exe_suffix}"
    returncode, _ = cmd_exec(
        [str(benchmark_app), "-d=CPU", f"-m={model}", "-niter=1", "-nireq=1"],
        env=get_openvino_environment(install_prefix),
    )
    assert returncode == 0, f"Command exited with non-zero status {returncode}"
