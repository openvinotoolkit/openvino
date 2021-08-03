#!/usr/bin/env python3

# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

""" Common utilities for working with paths
"""

import os
from pathlib import Path

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
            'Windows': Path('deployment_tools/inference_engine/bin/intel64/Release/inference_engine_transformations.dll'),
            'Linux': Path('deployment_tools/inference_engine/lib/intel64/libinference_engine_transformations.so')},
        'MKLDNNPlugin': {
            'Windows': Path('deployment_tools/inference_engine/bin/intel64/Release/MKLDNNPlugin.dll'),
            'Linux': Path('deployment_tools/inference_engine/lib/intel64/libMKLDNNPlugin.so')},
        'ngraph': {
            'Windows': Path('deployment_tools/ngraph/lib/ngraph.dll'),
            'Linux': Path('deployment_tools/ngraph/lib/libngraph.so')}
                }
    return all_libs[lib_name][os_name]
