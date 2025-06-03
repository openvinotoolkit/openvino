#!/usr/bin/env python3
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

""" Utility functions for work with json test configuration file.
"""
import os
import json
import multiprocessing
import sys
from inspect import getsourcefile
from pathlib import Path

from install_pkg import get_openvino_environment  # pylint: disable=import-error
from path_utils import get_lib_path  # pylint: disable=import-error
from proc_utils import cmd_exec  # pylint: disable=import-error

SESSION_INFO_FILE = "cc_tests.json"
infer_tool = str((Path(getsourcefile(lambda: 0)) / ".." / "tools" / "infer_tool.py").resolve())


def validate_path_arg(path: Path, is_dir=False):
    """"Check if path argument is correct."""
    if not path.exists():
        raise ValueError(f"{path} does not exist.")

    if is_dir and not path.is_dir():
        raise ValueError(f"{path} is not a directory.")

    return path


def get_lib_sizes(path, libraries):
    """ Function for getting lib sizes by lib names
    """
    assert Path.exists(path), f'Directory {path} isn\'t created'
    result = {}
    error_lib = []
    for lib in libraries:
        try:
            result[lib] = Path(path).joinpath(get_lib_path(lib)).stat().st_size
        except FileNotFoundError as error:
            error_lib.append(str(error))
    assert len(error_lib) == 0, 'Following libraries couldn\'t be found: \n{}'.format(
        '\n'.join(error_lib))
    return result


def read_session_info(path: Path = Path(getsourcefile(lambda: 0)).parent / SESSION_INFO_FILE):
    with open(path, 'r') as json_file:
        cc_tests_ids = json.load(json_file)
    return cc_tests_ids


def write_session_info(path: Path = Path(getsourcefile(lambda: 0)).parent / SESSION_INFO_FILE,
                       data: dict = None):
    with open(path, "w") as json_file:
        json.dump(data, json_file, indent=4)


def run_infer(models, out_dir, install_dir):
    """ Function running inference
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    return_code, output = cmd_exec(
        [sys.executable,
         infer_tool,
         "-d=CPU",
         *[f"-m={model}" for model in models],
         f"-r={out_dir}"
         ],
        env=get_openvino_environment(install_dir),
    )
    return return_code, output


def make_build(openvino_root_dir, build_dir, install_dir, build_target: dict = None, cmake_additional_args=None,
               log=None):
    """Parametrized build and install OpenVINO package."""
    additional_args_line = " ".join(cmake_additional_args) + " " if cmake_additional_args else ""
    build_target_arg_line = [f"cmake --build {build_target[target]} --target {target} && " for target in
                             build_target.keys()] if build_target else ""
    nproc = multiprocessing.cpu_count()
    cmd = (
        f"cmake -DENABLE_PROFILING_ITT=ON -DCMAKE_BUILD_TYPE=Release "
        f"-DPython3_EXECUTABLE={sys.executable} {additional_args_line}"
        f"-S {openvino_root_dir} -B {build_dir} && "
        f"cmake --build {build_dir} -j{nproc} && "
        f"{' '.join(build_target_arg_line)}"
        f"cmake --install {build_dir} --prefix {install_dir}"
    )
    return cmd_exec([cmd], shell=True, log=log)
