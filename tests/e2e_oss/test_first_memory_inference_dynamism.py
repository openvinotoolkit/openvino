"""Main entry-point to run E2E dynamism memory tests.

Default run:
$ pytest test_first_memory_inference_dynamism.py
Options[*]:
--modules            Paths to tests
--env_conf           Path to environment config
--test_conf          Path to test config
--dynamism_type      To specify dynamism type to test
--infer_binary_path  Path to memtest_infer binary file

[*] For more information see test_first_time_inference_dynamism.py
"""

import logging as log
import os
import sys

import pytest
import yaml

from e2e_oss.plugins.first_inference_tests.common_utils import get_redef_ir, get_reformat_shapes, get_trend
from e2e_oss.plugins.first_inference_tests.memorytests_utils import run_memorytest, get_compared_memory_results
from e2e_oss.plugins.first_inference_tests.timetests_utils import get_compared_with_refs_results

pytest_plugins = 'e2e_oss.plugins.reshape_tests.conftest'

# Define a number of iterations to run executable and aggregate results
NITER = 5
# Define a threshold for reference and status diff
REF_THRESHOLD_BOTTOM = 0.95
REF_THRESHOLD_UPPER = 1.05


def test_first_memory_inference_dynamism(instance, configuration, infer_binary_path, dynamism_type, prepare_test_info):
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.DEBUG, stream=sys.stdout)
    log.info(f"Running {instance.test_id} test")

    ir_input_shapes = configuration.shapes

    # Name of tests group
    prepare_test_info['pytestEntrypoint'] = 'Dynamism: first memory inference'
    prepare_test_info['inputsize'] = "__".join([f"{i}_{v}" for i, v in ir_input_shapes.items()])
    prepare_test_info['dynamismType'] = configuration.dynamism_type

    test_name = instance.__class__.__name__

    ir_input_shapes = configuration.shapes

    ir_path = get_redef_ir(instance, test_name, log)

    log.info("Starting memory measuring for Static IR")
    static_aggr_stats = run_memorytest(
        {"executable": infer_binary_path, "model": ir_path, "device": instance.device, "niter": NITER})
    log.info(static_aggr_stats)

    log.info("Starting memory measuring for Dynamic IR.")
    dynamic_aggr_stats = run_memorytest(
        {"executable": infer_binary_path, "model": ir_path, "device": instance.device, "niter": NITER,
         "reshape_shapes": get_reformat_shapes(ir_input_shapes)})
    log.info(dynamic_aggr_stats)

    status = get_compared_memory_results(static_aggr_stats, dynamic_aggr_stats)

    log.info(f"Difference between results (Dynamic / Static) as a percentage: ")
    yaml_status = yaml.dump(status, sort_keys=False)
    log.info(yaml_status)

    # Reference comparison
    ref_status_name = f"{test_name}"
    for name, shapes in ir_input_shapes.items():
        ref_status_name += f"_{name}_{shapes}"
    common_path_refs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plugins", "first_inference_tests",
                                    ".automation", f"references_{dynamism_type}")
    config_name = f"memory_test_references_config_{instance.device}.yml"
    config_path = os.path.join(common_path_refs, config_name)
    assert os.path.exists(config_path), f"Reference file {config_path} doesn't exists"

    with open(config_path, "r") as config_file:
        ref_content = yaml.safe_load(config_file)

    if ref_status_name in ref_content:
        log.info(f"Difference between current and reference results: ")
        no_degradation_per_metrics = []
        if not ref_content[ref_status_name]:
            raise ValueError(f"Reference results for {ref_status_name} were not found!")
        ref_val = get_compared_with_refs_results(ref_content[ref_status_name], status)
        catch_error = ref_val.get("ErrorOutput", None)
        assert not catch_error, f"Reference results for {ref_status_name} contain error string {catch_error}"
        for metric_name, ref_compared_res in ref_val.items():
            for memory_type, memory_type_stats in ref_compared_res.items():
                ratio = ref_compared_res[memory_type]['ratio']
                trend = get_trend(ratio, REF_THRESHOLD_BOTTOM, REF_THRESHOLD_UPPER)
                if trend == "Degradation":
                    no_degradation_per_metrics.append(False)
                else:
                    no_degradation_per_metrics.append(True)
                log.info(f"{metric_name} | {memory_type} | "
                         f"reference_value: {ref_compared_res[memory_type]['reference_value']} |"
                         f"current_value: {ref_compared_res[memory_type]['current_value']} |"
                         f"ratio: {ratio} |"
                         f"trend: {trend} |")

        assert all(no_degradation_per_metrics), "Some degradation has found in metrics. Please refer to log!"
    else:
        raise ValueError(f"Test name: {ref_status_name} \nwas not found in references: {config_path}!")
