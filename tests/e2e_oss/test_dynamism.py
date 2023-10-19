"""Main entry-point to run E2E dynamism tests.

Default run:
$ pytest test_dynamism.py
Options[*]:
--modules       Paths to tests
--env_conf      Path to environment config
--test_conf     Path to test config
--dynamism_type To specify dynamism type to test

To run specified tests, corresponding test classes must have
input_descriptor static field (test_reshape.py has description for this)

Dynamism_type key is also required and has two possible values:
1. "negative_ones":
      reference pipeline: e2e pipeline "as is"
      test pipeline: pipeline with '-1's for dimensions which was set in input_descriptor
      and ie.reshape for them (for example: 'data': [-1, 3, 224, 224])

2. "range_values":
      reference pipeline: e2e pipeline "as is"
      test pipeline: pipeline with [default_shape[i], input_descriptor_shape[i]]'s for dimensions
      which was set in input_descriptor and ie.reshape for them (for example: 'data': [[1, 10], 3, 224, 224])

[*] For more information see conftest.py
"""

import logging as log
import sys
from pathlib import Path

from e2e_oss.utils.modify_configs import dynamism_config, ie_reshape_config
from e2e_oss.utils.reshape_tests_utils import compare
from e2e_oss.utils.test_utils import set_infer_precision_hint, check_mo_precision, timestamp, get_static_shape
from tests.utils.e2e.common.pipeline import Pipeline

pytest_plugins = ('e2e_oss.plugins.reshape_tests.conftest',)


def test_dynamism(instance, configuration, prepare_test_info, copy_input_files, inference_precision_hint,
                  use_mo_cmd_tool):
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.DEBUG, stream=sys.stdout)
    log.info("Running {test_id} test".format(test_id=instance.test_id))

    test_name = instance.__class__.__name__
    instance.prepare_prerequisites()
    instance_ie_pipeline = instance.ie_pipeline

    instance_ie_pipeline = set_infer_precision_hint(instance, instance_ie_pipeline, inference_precision_hint)

    ref_pipeline = instance_ie_pipeline

    consecutive_infer = configuration.consecutive_infer
    ir_input_shapes = configuration.shapes
    default_shapes = configuration.default_shapes
    changed_values = configuration.changed_values
    layout = configuration.layout
    changed_dims = configuration.changed_dims

    # Name of tests group
    prepare_test_info['pytestEntrypoint'] = 'E2E: Dynamism'
    prepare_test_info['inputsize'] = "__".join([f"{i}_{v}" for i, v in ir_input_shapes.items()])
    prepare_test_info['dynamismType'] = configuration.dynamism_type

    xml_file = None
    bin_file = None

    if instance_ie_pipeline.get('postprocess'):
        del instance_ie_pipeline['postprocess']

    check_mo_precision(instance_ie_pipeline)
    dynamic_config = dynamism_config(instance_ie_pipeline, ir_input_shapes, test_name, default_shapes, changed_values,
                                     layout, changed_dims, consecutive_infer)

    try:
        instance_dynamic_pipeline = Pipeline(dynamic_config)
        log.info('Executing Dynamic reshape pipeline for {}'.format(test_name))
        instance_dynamic_pipeline.run()

    except Exception as err:
        log.error(err)
        raise Exception(f"{timestamp()} Dynamic reshape pipeline failed") from err

    ref_pipelines = []
    # first inference
    first_ref_pipeline = ie_reshape_config(ref_pipeline, default_shapes, test_name)

    if use_mo_cmd_tool:
        xml_file = Path(instance_dynamic_pipeline.details.xml)
        bin_file = xml_file.with_suffix('.bin')
        if 'pregenerated' not in first_ref_pipeline['get_ir']:
            first_ref_pipeline['get_ir'] = {"pregenerated": {"xml": xml_file, "bin": bin_file}}
            log.info(f"IR from first pipeline will be used, path to folder with IR: \n{xml_file.parent}")

    first_ref_pipeline = Pipeline(first_ref_pipeline)
    try:
        log.info('Executing Static reshape pipeline for {}'.format(test_name))
        first_ref_pipeline.run()
    except Exception as err:
        raise Exception(f"{timestamp()} Static reshape pipeline failed") from err
    ref_pipelines.append(first_ref_pipeline)

    if consecutive_infer:
        # second inference
        new_static_input_data_shapes = get_static_shape(default_shapes, changed_values, layout, changed_dims)
        second_ref_pipeline = ie_reshape_config(ref_pipeline, new_static_input_data_shapes, test_name)
        if use_mo_cmd_tool:
            if 'pregenerated' not in second_ref_pipeline['get_ir']:
                second_ref_pipeline['get_ir'] = {"pregenerated": {"xml": xml_file, "bin": bin_file}}
                log.info(f"IR from first pipeline will be used, path to folder with IR: \n{xml_file.parent}")

        second_ref_pipeline = Pipeline(second_ref_pipeline)

        try:
            log.info('Executing Static reshape pipeline for {}'.format(test_name))
            second_ref_pipeline.run()
        except Exception as err:
            raise Exception(f"{timestamp()} Static reshape pipeline failed") from err
        ref_pipelines.append(second_ref_pipeline)
    status = compare(instance, ref_pipelines, instance_dynamic_pipeline)
    assert status, 'inferred model results != reference results'
