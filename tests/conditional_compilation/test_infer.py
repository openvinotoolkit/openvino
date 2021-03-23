#!/usr/bin/env python3
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

""" Test inference with conditional compiled binaries.
"""

import sys
import os
from inspect import getsourcefile
from pathlib import Path
from proc_utils import cmd_exec  # pylint: disable=import-error
from install_pkg import get_openvino_environment  # pylint: disable=import-error


def run_infer(artifacts, test_id, model, out):
    install_prefix = artifacts / test_id / "install_pkg"
    returncode, output = cmd_exec(
        [sys.executable, str((Path(getsourcefile(lambda: 0)) / ".." / "tools" / "infer_tool.py").resolve()),
         "-d=CPU", f"-m={model}", f"-r={out}"],
        env=get_openvino_environment(install_prefix),
    )
    return returncode, output


def test_infer(test_id, model, artifacts, openvino_root_dir, openvino_cc):
    """ Test inference with conditional compiled binaries
    """
    tmp = os.environ["PATH"]
    out = artifacts / test_id
    os.environ["PATH"] = tmp.replace(os.path.join(*list(openvino_cc.parts)),
                                     os.path.join(*list(openvino_root_dir.parts)))
    returncode, output = run_infer(artifacts, test_id, model, out)
    assert returncode == 0, f"Command exited with non-zero status {returncode}:\n {output}"
    tmp = os.environ["PATH"]
    os.environ["PATH"] = tmp.replace(os.path.join(*list(openvino_root_dir.parts)),
                                     os.path.join(*list(openvino_cc.parts)))
    returncode, output = run_infer(artifacts, test_id, model, f"{out}_cc")
    assert returncode == 0, f"Command exited with non-zero status {returncode}:\n {output}"
