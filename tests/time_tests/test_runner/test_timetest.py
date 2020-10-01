# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Main entry-point to run timetests tests.

Default run:
$ pytest test_timetest.py

Options[*]:
--test_conf     Path to test config
--exe           Path to timetest binary to execute
--niter         Number of times to run executable

[*] For more information see conftest.py
"""

from pathlib import Path
import logging

from scripts.run_timetest import run_timetest


def test_timetest(instance, executable, niter):
    """Parameterized test.

    :param instance: test instance
    :param executable: timetest executable to run
    :param niter: number of times to run executable
    """
    # Prepare model to get model_path
    model_path = instance["model"].get("path")
    assert model_path, "Model path is empty"

    # Run executable
    exe_args = {
        "executable": Path(executable),
        "model": Path(model_path),
        "device": instance["device"]["name"],
        "niter": niter
    }
    retcode, aggr_stats = run_timetest(exe_args, log=logging)
    assert retcode == 0, "Run of executable failed"

    instance["results"] = aggr_stats    # append values to report to DB
