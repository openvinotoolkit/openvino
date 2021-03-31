#!/usr/bin/env python3
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

""" Utility functions for work with json test configuration file.
"""
import json

from inspect import getsourcefile
from pathlib import Path


SESSION_INFO_FILE = "cc_tests.json"


def read_session_info(path: Path = Path(getsourcefile(lambda: 0)).parent / SESSION_INFO_FILE):
    with open(path, 'r') as json_file:
        cc_tests_ids = json.load(json_file)
    return cc_tests_ids


def write_session_info(path: Path = Path(getsourcefile(lambda: 0)).parent / SESSION_INFO_FILE,
                       data: dict = None):
    with open(path, "w") as json_file:
        json.dump(data, json_file, indent=4)
