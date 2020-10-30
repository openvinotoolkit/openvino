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
import os
import shutil

from scripts.run_timetest import run_timetest
from test_runner.utils import expand_env_vars

REFS_FACTOR = 1.2      # 120%


def test_timetest(instance, executable, niter, cl_cache_dir, test_info, temp_dir, validate_test_case):
    """Parameterized test.

    :param instance: test instance. Should not be changed during test run
    :param executable: timetest executable to run
    :param niter: number of times to run executable
    :param cl_cache_dir: directory to store OpenCL cache
    :param test_info: custom `test_info` field of built-in `request` pytest fixture
    :param temp_dir: path to a temporary directory. Will be cleaned up after test run
    :param validate_test_case: custom pytest fixture. Should be declared as test argument to be enabled
    """
    # Prepare model to get model_path
    model_path = instance["model"].get("path")
    assert model_path, "Model path is empty"
    model_path = Path(expand_env_vars(model_path))

    # Copy model to a local temporary directory
    model_dir = temp_dir / "model"
    shutil.copytree(model_path.parent, model_dir)
    model_path = model_dir / model_path.name

    # Run executable
    exe_args = {
        "executable": Path(executable),
        "model": Path(model_path),
        "device": instance["device"]["name"],
        "niter": niter
    }
    if exe_args["device"] == "GPU":
        # Generate cl_cache via additional timetest run
        _exe_args = exe_args.copy()
        _exe_args["niter"] = 1
        logging.info("Run timetest once to generate cl_cache to {}".format(cl_cache_dir))
        run_timetest(_exe_args, log=logging)
        assert os.listdir(cl_cache_dir), "cl_cache isn't generated"

    retcode, aggr_stats = run_timetest(exe_args, log=logging)
    assert retcode == 0, "Run of executable failed"

    # Add timetest results to submit to database and save in new test conf as references
    test_info["results"] = aggr_stats

    # Compare with references
    comparison_status = 0
    for step_name, references in instance["references"].items():
        for metric, reference_val in references.items():
            if aggr_stats[step_name][metric] > reference_val * REFS_FACTOR:
                logging.error("Comparison failed for '{}' step for '{}' metric. Reference: {}. Current values: {}"
                              .format(step_name, metric, reference_val, aggr_stats[step_name][metric]))
                comparison_status = 1
            else:
                logging.info("Comparison passed for '{}' step for '{}' metric. Reference: {}. Current values: {}"
                             .format(step_name, metric, reference_val, aggr_stats[step_name][metric]))

    assert comparison_status == 0, "Comparison with references failed"

