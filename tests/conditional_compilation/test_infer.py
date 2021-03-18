#!/usr/bin/env python3
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

""" Test inference with conditional compiled binaries.
"""

import sys
import os
from proc_utils import cmd_exec  # pylint: disable=import-error
from install_pkg import get_openvino_environment  # pylint: disable=import-error


def run_infer(artifacts, test_id, model, infer_tool, out):
    install_prefix = artifacts / test_id / "install_pkg"
    returncode, output = cmd_exec(
        [sys.executable, str(infer_tool), "-d=CPU", f"-m={model}", f"-r={out}"],
        env=get_openvino_environment(install_prefix),
    )
    return returncode, output


def test_infer(test_id, model, artifacts, infer_tool,  install_dir, install_cc_dir):
    """ Test inference with conditional compiled binaries
    """
    tmp = os.environ["PATH"]
    out = artifacts / test_id
    os.environ["PATH"] = tmp.replace(os.path.join(*list(install_cc_dir.parts)), os.path.join(*list(install_dir.parts)))
    returncode, output = run_infer(artifacts, test_id, model, infer_tool, out)
    assert returncode == 0, f"Command exited with non-zero status {returncode}:\n {output}"
    tmp = os.environ["PATH"]
    os.environ["PATH"] = tmp.replace(os.path.join(*list(install_dir.parts)), os.path.join(*list(install_cc_dir.parts)))
    returncode, output = run_infer(artifacts, test_id, model, infer_tool, f"{out}_cc")
    assert returncode == 0, f"Command exited with non-zero status {returncode}:\n {output}"
