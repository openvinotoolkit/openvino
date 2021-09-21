#!/usr/bin/env python3

# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

""" Common utilities for working with paths
"""

import argparse
import os
import sys
from pathlib import Path

# add utils folder to imports
UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, str(UTILS_DIR))

from platform_utils import get_os_name


def expand_env_vars(obj):
    """Expand environment variables in provided object."""

    if isinstance(obj, list):
        for i, value in enumerate(obj):
            obj[i] = expand_env_vars(value)
    elif isinstance(obj, dict):
        for name, value in obj.items():
            obj[name] = expand_env_vars(value)
    else:
        obj = os.path.expandvars(obj)
    return obj


def get_lib_path(lib_name):
    """Function for getting absolute path in OpenVINO directory to specific lib"""
    os_name = get_os_name()
    all_libs = {
        'inference_engine_transformations': {
            'Windows': Path('runtime/bin/intel64/Release/inference_engine_transformations.dll'),
            'Linux': Path('runtime/lib/intel64/libinference_engine_transformations.so')},
        'MKLDNNPlugin': {
            'Windows': Path('runtime/bin/intel64/Release/MKLDNNPlugin.dll'),
            'Linux': Path('runtime/lib/intel64/libMKLDNNPlugin.so')},
        'ngraph': {
            'Windows': Path('runtime/bin/intel64/Release/ngraph.dll'),
            'Linux': Path('runtime/lib/intel64/libngraph.so')}
                }
    return all_libs[lib_name][os_name]


def check_positive_int(val):
    """Check argsparse argument is positive integer and return it"""
    value = int(val)
    if value < 1:
        msg = "%r is less than 1" % val
        raise argparse.ArgumentTypeError(msg)
    return value
