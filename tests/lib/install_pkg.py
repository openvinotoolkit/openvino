#!/usr/bin/env python3
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

""" Common utilities for OpenVINO install package.
"""
import sys
from pathlib import Path
from proc_utils import get_env_from  # pylint: disable=import-error


def get_openvino_environment(install_prefix: Path):
    """ Get OpenVINO environment variables
    """
    script = "setupvars.bat" if sys.platform == "win32" else "setupvars.sh"
    return get_env_from(install_prefix / "bin" / script)
