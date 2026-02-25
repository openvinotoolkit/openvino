#!/usr/bin/env python3

# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

""" Common utilities for OpenVINO install package.
"""
import errno
import os
from pathlib import Path
import subprocess
import sys


def get_openvino_environment(install_prefix: Path):
    """ Get OpenVINO environment variables
    """
    if sys.platform == "win32":
        script = install_prefix / "setupvars.bat"
        cmd = f"{script} && set"
    else:
        script = install_prefix / "setupvars.sh"
        # setupvars.sh is not compatible with /bin/sh. Using bash.
        cmd = f'bash -c ". {script} && env"'

    if not os.path.exists(str(script)):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(script))

    env = {}
    dump = subprocess.check_output(cmd, shell=True, universal_newlines=True).strip()
    for line in dump.split("\n"):
        # split by first '='
        pair = [str(val).strip() for val in line.split("=", 1)]
        if len(pair) > 1 and pair[0]:  # ignore invalid entries
            env[pair[0]] = pair[1]
    return env
