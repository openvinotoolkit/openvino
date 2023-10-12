"""Runner for comparing IR inference results to ONNX model inference results (through ONNX importer)
using eltwise comparison (the most strict one).

Default run:
$ pytest test.py

Options[*]:
--modules       Paths to tests
--env_conf      Path to environment config
--test_conf     Path to test config

[*] For more information see conftest.py
"""
import logging as log
import re
import sys
import time
from pathlib import Path

from e2e_oss._utils.test_utils import get_ir_tag, read_irs_mapping_file, set_infer_precision_hint
from utils.e2e.common.pipeline import Pipeline
from utils.e2e.comparator.container import ComparatorsContainer
from utils.parsers import pipeline_cfg_to_string
from utils.path_utils import prepend_with_env_path

pytest_plugins = ('e2e_oss.plugins.e2e_test.conftest',)


def test_compare_onnx_ir(instance, pregen_irs, record_property, use_mo_legacy_frontend, use_mo_new_frontend,
                         prepare_test_info, inference_precision_hint):
    """Parameterized test.

    :param instance: test instance
    """
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.DEBUG, stream=sys.stdout)

    # Name of tests group
    prepare_test_info['pytestEntrypoint'] = 'Compare IR vs ONNX model inference results'

    log.info("Running {test_id} test".format(test_id=instance.test_id))
    instance.prepare_prerequisites()
    instance_ie_pipeline = instance.ie_pipeline

    ir_version = "v11"

    if next(iter(instance_ie_pipeline["get_ir"])) == "mo" and use_mo_legacy_frontend:
        instance_ie_pipeline["get_ir"]["mo"]["additional_args"].update({"use_legacy_frontend": True})
        prepare_test_info['pytestEntrypoint'] = 'Compare IR vs ONNX model inference results: legacy frontend'
        if not instance.api_2:
            prepare_test_info['pytestEntrypoint'] = 'Compare IR vs ONNX model inference results: legacy frontend old API'

    if next(iter(instance_ie_pipeline["get_ir"])) == "mo" and use_mo_new_frontend:
        instance_ie_pipeline["get_ir"]["mo"]["additional_args"].update({"use_new_frontend": True})

    if instance_ie_pipeline.get('infer'):
        instance_ie_pipeline = set_infer_precision_hint(instance, instance_ie_pipeline, inference_precision_hint)

    if pregen_irs and "get_ir" in instance_ie_pipeline:
        try:
            log.info("Searching pre-generated IR in IR's mapping: {} ...".format(pregen_irs))
            irs_mapping = read_irs_mapping_file(pregen_irs)
            ir_tag = get_ir_tag(instance.__class__.__name__, ir_version, instance.precision,
                                instance.batch, instance.required_params.get("sequence_length", None),
                                use_mo_legacy_frontend, use_mo_new_frontend)
            if ir_tag not in irs_mapping:
                log.warning("IR with tag '{}' not found in IRs mapping. "
                            "IR will be generated in runtime ...".format(ir_tag))
            else:
                log.info("Found pre-generated IR entry in IRs mapping: {}.\nTrying to reuse it ..."
                         .format({ir_tag: irs_mapping[ir_tag]}))
                pregen_ir_status, mo_log, xml, bin = irs_mapping[ir_tag]
                if not pregen_ir_status:
                    log.error('IR pre-generation failed. IR will be generated in runtime ...')
                else:
                    if not mo_log:
                        log.warning('IR was collected successfully, but MO log was not saved.')
                    else:
                        with open(mo_log, "r") as file:
                            mo_output = file.read()
                            log.info("Model Optimizer output:\n{output}".format(output=mo_output))
                    if not (Path(xml).exists() and Path(bin).exists()):
                        log.error("One of IR's .xml or .bin files not found. IR will be generated in runtime ...")
                    else:
                        ir_http_path = "http://{}".format(re.sub(r"^[/|\\]+", "", xml).replace("\\", "/"))
                        record_property("ir_link", ir_http_path)
                        instance_ie_pipeline["get_ir"] = {"pregenerated": {"xml": xml, "bin": bin}}
        except Exception as e:
            log.error("Search of pre-generated IR failed with error: {err}. IR will be generated in runtime ..."
                      .format(err=e))

    ie_pipeline_original = Pipeline(instance_ie_pipeline)
    del instance_ie_pipeline['get_ir']
    orig_net_modifiers = instance_ie_pipeline["infer"].get("ie_sync", {}).get("network_modifiers", {})
    instance_ie_pipeline.update({'infer': {'ie_sync': {'model_path': prepend_with_env_path(instance.model_env_key,
                                                                                           instance.model),
                                                       'device': instance.device,
                                                       'network_modifiers': orig_net_modifiers}}})
    if getattr(instance, 'reshape_map', None):
        instance_ie_pipeline['infer']['ie_sync']['network_modifiers'] = \
            {'reshape': {'shapes': instance.reshape_map}}
    ie_pipeline_onnx = Pipeline(instance_ie_pipeline)

    log.debug("Test scenario:")
    log.debug("Original IR inference pipeline:\n{}".format(pipeline_cfg_to_string(ie_pipeline_original._config)))
    log.debug("ONNX model inference pipeline:\n{}".format(pipeline_cfg_to_string(ie_pipeline_onnx._config)))
    log.info("Running IR inference pipeline:")
    ie_pipeline_original.run()

    log.info("Running ONNX model inference pipeline:")
    ie_start = time.time()
    ie_pipeline_onnx.run()
    ie_end = time.time()
    log.info("ONNX model infer time: {ir_time}".format(ir_time=str(ie_end - ie_start)))

    # for classification and eltwise cases
    if 'eltwise' in instance.comparators.keys():
        eltwise_comp_attrs = instance.comparators['eltwise']
        eltwise_comp_attrs['ignore_results'] = False

    else:
        eltwise_comp_attrs = {'device': instance.device, 'a_eps': None, 'r_eps': None,
                              'precision': instance.precision,
                              'target_layers': None, 'ignore_results': False}
    instance.comparators = {'eltwise': eltwise_comp_attrs}
    comparators = ComparatorsContainer(
        config=instance.comparators,
        infer_result=ie_pipeline_onnx.fetch_results(),
        reference=ie_pipeline_original.fetch_results(),
        result_aligner=getattr(instance, 'align_results', None))
    log.info("Running comparators:")
    comparators.apply_postprocessors()
    comparators.apply_all()
    status = comparators.report_statuses()
    assert status, "inferred model results != reference results"
