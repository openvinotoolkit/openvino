#!/usr/bin/env python3
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

""" Common utilities for working with paths
"""

import os


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
