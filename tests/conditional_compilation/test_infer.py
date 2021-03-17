#!/usr/bin/env python3
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

""" Test inference with conditional compiled binaries.
"""
import sys, os
from proc_utils import cmd_exec  # pylint: disable=import-error
from install_pkg import get_openvino_environment  # pylint: disable=import-error


def test_infer(test_id, model, artifacts, infer_tool,  install_dir, install_cc_dir):
    """ Test inference with conditional compiled binaries
    """
    tmp = os.environ["PATH"]
    tmp = tmp.replace(os.path.join(*list(install_cc_dir.parts)), os.path.join(*list(install_dir.parts)))
    os.environ["PATH"] = tmp
    model_result_path = os.path.join(*list(artifacts.parts), model.stem)
    returncode, output = cmd_exec(
        [sys.executable, str(infer_tool), "-d=CPU", f"-m={model}", "-d=CPU",
            f"-r={model_result_path}"],
    )
    print(output)
    assert returncode == 0, f"Command exited with non-zero status {returncode}:\n {output}"
    tmp = os.environ["PATH"]
    tmp = tmp.replace(os.path.join(*list(install_dir.parts)), os.path.join(*list(install_cc_dir.parts)))
    os.environ["PATH"] = tmp
    model_result_path = os.path.join(*list(artifacts.parts), f"{model.stem}_cc")
    returncode, output = cmd_exec(
        [sys.executable, str(infer_tool), "-d=CPU", f"-m={model}", "-d=CPU",
         f"-r={model_result_path}"],
    )
    print(output)
    assert returncode == 0, f"Command exited with non-zero status {returncode}:\n {output}"