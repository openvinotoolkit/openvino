# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Main entry-point to run tests tests.
Default run:
$ pytest test_test.py
Options[*]:
--test_conf     Path to test config
--exe           Path to test binary to execute
--niter         Number of times to run executable
[*] For more information see conftest.py
"""

import logging
import os
import shutil
import sys
from pathlib import Path

# add utils folder to imports
UTILS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "utils")
sys.path.insert(0, str(UTILS_DIR))

from path_utils import expand_env_vars

MEMORY_TESTS_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(MEMORY_TESTS_DIR)

from scripts.run_memorytest import run_memorytest
from test_runner.utils import compare_with_references


def test(instance, executable, niter, temp_dir, omz_models_conversion, validate_test_case, prepare_db_info):
    """Parameterized test.
    :param instance: test instance. Should not be changed during test run
    :param executable: test executable to run
    :param niter: number of times to run executable
    :param temp_dir: path to a temporary directory. Will be cleaned up after test run
    :param validate_test_case: custom pytest fixture. Should be declared as test argument to be enabled
    :param prepare_db_info: custom pytest fixture. Should be declared as test argument to be enabled
    :param omz_models_conversion: custom pytest fixture. Should be declared as test argument to be enabled
    """
    # Prepare model to get model_path
    model_path = ''
    cache_model_path = instance["instance"]["model"].get("cache_path")
    irs_model_path = instance["instance"]["model"].get("irs_out_path")

    if os.path.isfile(irs_model_path):
        model_path = irs_model_path
    elif os.path.isfile(cache_model_path):
        model_path = cache_model_path

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
        "device": instance["instance"]["device"]["name"],
        "niter": niter
    }
    retcode, msg, aggr_stats, raw_stats = run_memorytest(exe_args, log=logging)
    assert retcode == 0, f"Run of executable failed: {msg}"

    # Add test results to submit to database and save in new test conf as references
    instance["results"] = aggr_stats
    instance["raw_results"] = raw_stats

    # Compare with references
    metrics_comparator_status = compare_with_references(aggr_stats, instance["orig_instance"]["references"])
    assert metrics_comparator_status == 0, "Comparison with references failed"
