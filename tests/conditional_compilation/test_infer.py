#!/usr/bin/env python3
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

""" Test inference with conditional compiled binaries.
"""

import sys
from inspect import getsourcefile
from pathlib import Path
from proc_utils import cmd_exec  # pylint: disable=import-error
from install_pkg import get_openvino_environment  # pylint: disable=import-error


def run_infer(model, out, install_dir):
    """ Function running inference
    """

    returncode, output = cmd_exec(
        [sys.executable,
         str((Path(getsourcefile(lambda: 0)) / ".." / "tools" / "infer_tool.py").resolve()),
         "-d=CPU", f"-m={model}", f"-r={out}"
         ],
        env=get_openvino_environment(install_dir),
    )
    return returncode, output


def test_infer(test_id, model, artifacts, openvino_root_dir, openvino_cc):
    """ Test inference with conditional compiled binaries
    """
    out = artifacts / test_id
    returncode, output = run_infer(model, out, openvino_root_dir)
    assert returncode == 0, f"Command exited with non-zero status {returncode}:\n {output}"
    returncode, output = run_infer(model, f"{out}_cc", openvino_cc)
    assert returncode == 0, f"Command exited with non-zero status {returncode}:\n {output}"
