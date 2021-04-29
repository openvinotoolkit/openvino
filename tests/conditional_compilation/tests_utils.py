#!/usr/bin/env python3
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

""" Utility functions for work with json test configuration file.
"""
import os
import json
import sys
from inspect import getsourcefile
from pathlib import Path
from proc_utils import cmd_exec  # pylint: disable=import-error
from install_pkg import get_openvino_environment  # pylint: disable=import-error

SESSION_INFO_FILE = "cc_tests.json"
infer_tool = str((Path(getsourcefile(lambda: 0)) / ".." / "tools" / "infer_tool.py").resolve())


def read_session_info(path: Path = Path(getsourcefile(lambda: 0)).parent / SESSION_INFO_FILE):
    with open(path, 'r') as json_file:
        cc_tests_ids = json.load(json_file)
    return cc_tests_ids


def write_session_info(path: Path = Path(getsourcefile(lambda: 0)).parent / SESSION_INFO_FILE,
                       data: dict = None):
    with open(path, "w") as json_file:
        json.dump(data, json_file, indent=4)


def run_infer(model, out_file, install_dir):
    """ Function running inference
    """
    sys_executable = os.path.join(sys.prefix, 'python.exe') if sys.platform == "win32" \
        else os.path.join(sys.prefix, 'bin', 'python')
    returncode, output = cmd_exec(
        [sys_executable,
         infer_tool,
         "-d=CPU", f"-m={model}", f"-r={out_file}"
         ],
        env=get_openvino_environment(install_dir),
    )
    return returncode, output
