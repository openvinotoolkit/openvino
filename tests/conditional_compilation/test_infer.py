#!/usr/bin/env python3
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

""" Test inference with conditional compiled binaries.
"""

from proc_utils import cmd_exec  # pylint: disable=import-error


def test_infer(model, benchmark_app):
    """ Test inference with conditional compiled binaries
    """
    returncode, _ = cmd_exec(
        [str(benchmark_app), "-d=CPU", f"-m={model}", "-niter=1", "-nireq=1"]
    )
    assert returncode == 0, f"Command exited with non-zero status {returncode}"
