#!/usr/bin/env python3

# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

""" Test conditional compilation statistics collection.
"""

import glob
import os
import sys

from proc_utils import cmd_exec  # pylint: disable=import-error


def test_cc_collect(test_id, model, sea_runtool, benchmark_app, collector_dir, artifacts, test_info):
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
    return_code, output = cmd_exec(
        [
            sys.executable,
            str(sea_runtool),
            f"--output={out}",
            f"--bindir={collector_dir}",
            "--app_status",
            "!",
            str(benchmark_app),
            "-d=CPU",
            f"-m={model}",
            "-niter=1",
            "-nireq=1",
        ]
    )
    out_csv = glob.glob(f"{out}.pid*.csv")
    test_info["out_csv"] = out_csv

    assert return_code == 0, f"Command exited with non-zero status {return_code}:\n {output}"
    assert (len(out_csv) == 1), f'Multiple or none "{out}.pid*.csv" files'
