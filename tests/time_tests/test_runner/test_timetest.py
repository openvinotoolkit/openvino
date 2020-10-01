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
from test_runner.utils import expand_env_vars

REFS_FACTOR = 1.2      # 120%


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
        "model": Path(expand_env_vars(model_path)),
        "device": instance["device"]["name"],
        "niter": niter
    }
    retcode, aggr_stats = run_timetest(exe_args, log=logging)
    assert retcode == 0, "Run of executable failed"

    # Add timetest results to submit to database and save in new test conf as references
    instance["results"] = aggr_stats

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

