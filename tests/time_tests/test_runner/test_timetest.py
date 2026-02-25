# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

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
import sys

# add utils folder to imports
UTILS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "utils")
sys.path.insert(0, str(UTILS_DIR))

from path_utils import expand_env_vars

TIME_TESTS_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(TIME_TESTS_DIR)

from scripts.run_timetest import run_timetest

REFS_FACTOR = 1.2      # 120%


def test_timetest(instance, executable, niter, cl_cache_dir, model_cache, model_cache_dir,
                  test_info, temp_dir, validate_test_case, prepare_db_info):
    """Parameterized test.

    :param instance: test instance. Should not be changed during test run
    :param executable: timetest executable to run
    :param niter: number of times to run executable
    :param cl_cache_dir: directory to store OpenCL cache
    :param cpu_cache: flag to enable model CPU cache
    :param npu_compiler: flag to change NPU compiler type
    :param perf_hint: performance hint (optimize device for latency or throughput settings)
    :param model_cache_dir: directory to store OV model cache
    :param test_info: custom `test_info` field of built-in `request` pytest fixture
    :param temp_dir: path to a temporary directory. Will be cleaned up after test run
    :param validate_test_case: custom pytest fixture. Should be declared as test argument to be enabled
    :param prepare_db_info: custom pytest fixture. Should be declared as test argument to be enabled
    """
    # Prepare model to get model_path
    model_path = instance["model"].get("path")
    assert model_path, "Model path is empty"
    model_path = Path(expand_env_vars(model_path))

    # Prepare input precision from model configuration
    input_precision = instance["model"].get("input_precision")

    # Prepare output precision from model configuration
    output_precision = instance["model"].get("output_precision")

    # Copy model to a local temporary directory
    model_dir = temp_dir / "model"
    shutil.copytree(model_path.parent, model_dir)
    model_path = model_dir / model_path.name

    # Run executable
    exe_args = {
        "executable": Path(executable),
        "model": Path(model_path),
        "device": instance["device"]["name"],
        "niter": niter,
        "input_precision": input_precision,
        "output_precision": output_precision,
        "model_cache": model_cache,
    }
    logging.info("Run timetest once to generate any cache")
    retcode, msg, _, _, _ = run_timetest({**exe_args, "niter": 1}, log=logging)
    assert retcode == 0, f"Run of executable for warm up failed: {msg}"
    if cl_cache_dir:
        assert os.listdir(cl_cache_dir), "cl_cache isn't generated"
    if model_cache_dir:
        assert os.listdir(model_cache_dir), "model_cache isn't generated"

    retcode, msg, aggr_stats, raw_stats, logs = run_timetest(exe_args, log=logging)
    test_info["logs"] = "\n".join(logs)
    assert retcode == 0, f"Run of executable failed: {msg}"

    # Add timetest results to submit to database and save in new test conf as references
    test_info["results"] = aggr_stats
    test_info["raw_results"] = raw_stats
