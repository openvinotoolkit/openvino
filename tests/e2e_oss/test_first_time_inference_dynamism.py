"""Main entry-point to run FIL in dynamism pipelines.

Default run:
$ pytest test_first_time_inference_dynamism.py
Options[*]:
--modules            Paths to tests
--env_conf           Path to environment config
--test_conf          Path to test config
--dynamism_type      To specify dynamism type to test
--infer_binary_path  Path to timetest_infer binary file

How to interpret the results?

The result is a percentage of the dynamic and static result of the time tests.
A negative metric result indicates a faster dynamic result as opposed to a static one.
When comparing the two results, it is worth considering the sign of the resulting expression.
The test will be failed if the current result is 15 percent worse than the reference one. (see REF_THRESHOLD)

Example:
load_plugin | ref_value: 1.173 | cur_value: -18.677 | ref - cur: 19.85 (PASS, current result better than reference)
load_plugin | ref_value: -5.173 | cur_value: 10.677 | ref - cur: -15.85 (FAIL, current result worse than reference)

[*] For more information see test_dynamism.py
"""

import logging as log
import os
import sys

import pytest
import yaml

from tests.e2e_oss.plugins.first_inference_tests.common_utils import get_redef_ir, get_reformat_shapes
from tests.e2e_oss.plugins.first_inference_tests.memorytests_utils import get_compared_with_refs_results
from tests.e2e_oss.plugins.first_inference_tests.timetests_utils import run_timetest, get_compared_time_results

pytest_plugins = 'e2e_oss.plugins.reshape_tests.conftest'

# Define a number of iterations to run executable and aggregate results
NITER = 5
# Define a threshold for reference and status diff
REF_THRESHOLD = -15


def test_first_time_inference_dynamism(instance, configuration, infer_binary_path, dynamism_type, prepare_test_info):
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.DEBUG, stream=sys.stdout)
    log.info(f"Running {instance.test_id} test")

    # Name of tests group
    prepare_test_info['pytestEntrypoint'] = 'Dynamism: first time inference'

    test_name = instance.__class__.__name__

    ir_input_shapes = configuration.shapes

    ir_path = get_redef_ir(instance, test_name, log)

    log.info("Starting time measuring for Static IR")
    static_aggr_stats = run_timetest(
        {"executable": infer_binary_path, "model": ir_path, "device": instance.device, "niter": NITER})
    log.info(static_aggr_stats)

    log.info("Starting time measuring for Dynamic IR.")
    dynamic_aggr_stats = run_timetest(
        {"executable": infer_binary_path, "model": ir_path, "device": instance.device, "niter": NITER,
         "reshape_shapes": get_reformat_shapes(ir_input_shapes)})
    log.info(dynamic_aggr_stats)

    status = get_compared_time_results(static_aggr_stats, dynamic_aggr_stats)

    log.info(f"Difference between results (Dynamic / Static) as a percentage: ")
    for metric, value in status.items():
        log.info(f"{metric} | {value} %")

    # Reference comparison
    ref_status_name = f"{test_name}"
    for name, shapes in ir_input_shapes.items():
        ref_status_name += f"_{name}_{shapes}"

    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plugins", "first_inference_tests",
                               ".automation", f"references_{dynamism_type}", "time_test_references_config.yml")

    with open(config_path, "r") as config_file:
        ref_content = yaml.safe_load(config_file)

    if ref_status_name in ref_content:
        log.info(f"Difference between current and reference results: ")
        ref_val = get_compared_with_refs_results(ref_content[ref_status_name], status)
        if not ref_val:
            log.warning("Reference results were not found!")
        for metric_name, ref_compared_res in ref_val.items():
            log.info(f"{metric_name} | ref_value: {ref_compared_res['ref']} |"
                     f" cur_value: {ref_compared_res['cur']} |"
                     f" ref - cur: {ref_compared_res['ref - cur']}")
        for metric_name, ref_compared_res in ref_val.items():
            assert ref_compared_res['ref - cur'] >= REF_THRESHOLD, \
                f"FIL metric '{metric_name}' failed on threshold!"
    else:
        log.warning("Test name was not found in references!")
