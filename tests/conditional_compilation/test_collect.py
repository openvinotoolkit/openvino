#!/usr/bin/env python3

# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

""" Test conditional compilation statistics collection.
"""

import glob
import os
import sys
import pytest

from proc_utils import cmd_exec  # pylint: disable=import-error
from tests_utils import write_session_info, SESSION_INFO_FILE, infer_tool


@pytest.fixture(scope="function")
def test_info(request, pytestconfig):
    """Fixture function for getting the additional attributes of the current test."""
    setattr(request.node._request, "test_info", {})
    if not hasattr(pytestconfig, "session_info"):
        setattr(pytestconfig, "session_info", [])

    yield request.node._request.test_info

    pytestconfig.session_info.append(request.node._request.test_info)


@pytest.fixture(scope="session")
def save_session_info(pytestconfig, artifacts):
    """Fixture function for saving additional attributes to configuration file."""
    yield
    write_session_info(path=artifacts / SESSION_INFO_FILE, data=pytestconfig.session_info)


def test_cc_collect(test_id, model, sea_runtool, collector_dir, artifacts, test_info, save_session_info):
    """ Test conditional compilation statistics collection
    :param test_info: custom `test_info` field of built-in `request` pytest fixture.
                      contain a dictionary to store test metadata.
    """
    out = artifacts / test_id
    test_info["test_id"] = test_id
    # cleanup old data if any
    prev_result = glob.glob(f"{out}.pid*.csv")
    for path in prev_result:
        os.remove(path)
    # run use case
    sys_executable = os.path.join(sys.prefix, 'python.exe') if sys.platform == "win32" \
        else os.path.join(sys.prefix, 'bin', 'python')
    return_code, output = cmd_exec(
        [
            sys_executable,
            str(sea_runtool),
            f"--output={out}",
            f"--bindir={collector_dir}",
            "!",
            sys_executable,
            infer_tool,
            f"-m={model}",
            "-d=CPU",
            f"-r={out}",
        ]
    )
    out_csv = glob.glob(f"{out}.pid*.csv")
    test_info["out_csv"] = out_csv

    assert return_code == 0, f"Command exited with non-zero status {return_code}:\n {output}"
    assert (len(out_csv) == 1), f'Multiple or none "{out}.pid*.csv" files'
