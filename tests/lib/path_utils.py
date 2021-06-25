#!/usr/bin/env python3

# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

""" Common utilities for working with paths
"""

import os
import sys
from pathlib import Path


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


def get_os_name():
    """Function for getting OS name"""
    if sys.platform == "win32":
        os_name = 'Windows'
    else:
        os_name = 'Linux'
    return os_name


def get_lib_path(lib_name):
    """Function for getting absolute path in OpenVINO directory to specific lib"""
    os_name = get_os_name()
    all_libs = {
        'inference_engine_transformations': {
            'Windows': Path('runtime/bin/inference_engine_transformations.dll'),
            'Linux': Path('runtime/lib/libinference_engine_transformations.so')},
        'MKLDNNPlugin': {
            'Windows': Path('runtime/bin/MKLDNNPlugin.dll'),
            'Linux': Path('runtime/lib/libMKLDNNPlugin.so')},
        'ngraph': {
            'Windows': Path('runtime/bin/ngraph.dll'),
            'Linux': Path('runtime/lib/libngraph.so')}
                }
    return all_libs[lib_name][os_name]
