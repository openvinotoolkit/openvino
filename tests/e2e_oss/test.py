"""Main entry-point to run E2E OSS tests.

Default run:
$ pytest test.py

Options[*]:
--modules       Paths to tests
--env_conf      Path to environment config
--test_conf     Path to test config

[*] For more information see conftest.py
"""
# pylint:disable=invalid-name
import logging as log
import os
import re
from pathlib import Path
from shutil import rmtree
import yaml

from e2e_oss.common_utils.logger import get_logger
from e2e_oss._utils.modify_configs import get_original_model_importer_pipeline_config
from e2e_oss._utils.test_utils import log_timestamp, read_irs_mapping_file, get_ir_tag, check_mo_precision, \
    set_infer_precision_hint, remove_mo_args_oob, store_data_to_csv, timestamp
from e2e_oss.common_utils.parsers import pipeline_cfg_to_string
from e2e_oss.common_utils.test_utils import get_shapes_from_data
from utils.e2e.common.pipeline import Pipeline
from utils.e2e.comparator.container import ComparatorsContainer
from utils.e2e.env_tools import Environment

pytest_plugins = ('e2e_oss.plugins.e2e_test.conftest',)

log = get_logger(__name__)


def _test_run(instance, pregen_irs, ir_gen_time_csv_name, load_net_to_plug_time_csv_name, mem_usage_mo_csv_name,
              mem_usage_ie_csv_name, record_property, use_mo_legacy_frontend, use_mo_new_frontend, skip_mo_args,
              dynamic_inference, prepare_test_info, skip_ir_generation, copy_input_files, inference_precision_hint):
    """Parameterized test.

    :param instance: test instance
    :param pregen_irs: custom fixture. Provides path to a CSV-formatted file with IRs mapping
    :param ir_gen_time_csv_name: name for csv file with IR generation time
    :param load_net_to_plug_time_csv_name: name for csv file with load net to plugin time
    :param mem_usage_mo_csv_name: name for csv file with MO memory usage information
    :param mem_usage_ie_csv_name: name for csv file with IE memory usage information
    :param record_property: extra property for the calling test
    :param skip_mo_args: line with comma separated args that will be deleted from MO cmd line
    :param dynamic_inference: bool value to enable dynamic inference mode
    :param instance: test instance
    """
    api_2 = instance.api_2
    model_framework_name = instance.__class__.__name__.split('_')[0]
    # Name of tests group
    prepare_test_info['pytestEntrypoint'] = 'E2E: Base'
    if not api_2:
        prepare_test_info['pytestEntrypoint'] = 'E2E: Base old API'

    if use_mo_new_frontend:
        if 'PDPD' in model_framework_name:
            prepare_test_info['pytestEntrypoint'] = 'E2E: PDPD new FrontEnd'
            if not api_2:
                prepare_test_info['pytestEntrypoint'] = 'E2E: PDPD new FrontEnd with old API'

    if hasattr(instance, 'sequence_length'):
        if isinstance(instance.sequence_length, int):
            prepare_test_info['inputsize'] = f"sequence_length:{instance.sequence_length}"

    ir_version = "v11"

    log.info("Running {test_id} test".format(test_id=instance.test_id))
    instance.prepare_prerequisites()
    instance_ie_pipeline = instance.ie_pipeline
    instance_ref_pipeline = instance.ref_pipeline

    ref_pipeline = Pipeline(instance_ref_pipeline)
    log.debug("Test scenario:")
    log.debug("Reference Pipeline:\n{}".format(pipeline_cfg_to_string(ref_pipeline._config)))
    if ref_pipeline.steps:
        with log_timestamp('reference pipeline'):
            log.info("Running reference pipeline:")
            ref_pipeline.run()
    else:
        log.warning("Reference pipeline is empty, no results comparison will be performed")

    log.info("Running inference pipeline:")
    if pregen_irs and "get_ir" in instance_ie_pipeline:
        try:
            log.info("Searching pre-generated IR in IR's mapping: {} ...".format(pregen_irs))
            irs_mapping = read_irs_mapping_file(pregen_irs)
            instance.required_params = {"sequence_length": instance.sequence_length} if type(instance.sequence_length) == int else {}
            ir_tag = get_ir_tag(instance.__class__.__name__, ir_version, instance.precision,
                                instance.batch, instance.required_params.get("sequence_length", None),
                                use_mo_legacy_frontend, use_mo_new_frontend, skip_mo_args)
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
                        ir_http_path = "http://{}".format(re.sub(r"^[/|\\]+", "", str(xml)).replace("\\", "/"))
                        record_property("ir_link", ir_http_path)
                        instance_ie_pipeline["get_ir"] = {"pregenerated": {"xml": xml, "bin": bin}}
        except Exception as ex:
            log.error("Search of pre-generated IR failed with error: {err}."
                      " IR will be generated in runtime ...".format(err=ex))

    check_mo_precision(instance_ie_pipeline)

    if next(iter(instance_ie_pipeline["get_ir"])) == "mo" and instance.use_mo_cmd_tool:
        instance_ie_pipeline["get_ir"]["mo"].update({"use_mo_cmd_tool": instance.use_mo_cmd_tool})

    if next(iter(instance_ie_pipeline["get_ir"])) == "mo" and use_mo_new_frontend:
        instance_ie_pipeline["get_ir"]["mo"]["additional_args"].update({"use_new_frontend": True})

    if instance_ie_pipeline.get('infer'):
        instance_ie_pipeline = set_infer_precision_hint(instance, instance_ie_pipeline, inference_precision_hint)

    if next(iter(instance_ie_pipeline["get_ir"])) == "mo" and skip_mo_args:
        # Rename of tests group because "skip_mo_args" is other group of E2E tests
        prepare_test_info['pytestEntrypoint'] = 'E2E: OOB'
        mo_additional_args = instance_ie_pipeline['get_ir']['mo'].get('additional_args', {})
        if mo_additional_args:
            instance_ie_pipeline['get_ir']['mo']['additional_args'] = remove_mo_args_oob(skip_mo_args,
                                                                                         mo_additional_args,
                                                                                         instance_ie_pipeline)
        if dynamic_inference:
            if not instance_ie_pipeline.get('infer'):
                log.info("No infer step for this test")
            elif instance_ie_pipeline['infer'].get('ie_sync', {}).get('network_modifiers'):
                del instance_ie_pipeline['infer']['ie_sync']['network_modifiers']
            else:
                log.info('Network modifiers for were not found in this test')
        else:
            if instance_ie_pipeline.get('infer', {}).get('ie_sync'):
                input_pipeline = Pipeline({'read_input': instance_ie_pipeline.get('read_input', {}),
                                           'preprocess': instance_ie_pipeline.get('preprocess', {})})
                input_pipeline.run()
                shapes = get_shapes_from_data(input_pipeline.fetch_results())
                instance_ie_pipeline['infer']['ie_sync']['network_modifiers'] = {'reshape': {'shapes': shapes}}
            else:
                log.info("No infer step for this test")

    if skip_ir_generation:
        instance_ie_pipeline = get_original_model_importer_pipeline_config(instance_ie_pipeline)
        if 'ONNX' or 'Precollected' or 'Caffe2' or 'WinML' in model_framework_name:
            prepare_test_info['pytestEntrypoint'] = 'E2E: ONNX Without MO step'
        if 'PDPD' in model_framework_name:
            prepare_test_info['pytestEntrypoint'] = 'E2E: PDPD Without MO step'
        if 'TF' in model_framework_name:
            prepare_test_info['pytestEntrypoint'] = 'E2E: TF Without MO step'
        if not api_2:
            prepare_test_info['pytestEntrypoint'] += ' old API'

    ie_pipeline = Pipeline(instance_ie_pipeline)
    log.debug("Inference Pipeline:\n{}".format(pipeline_cfg_to_string(ie_pipeline._config)))
    ie_pipeline.run()

    if not skip_ir_generation and instance.use_mo_cmd_tool:
        instance.mapping_file_location = getattr(instance, "mapping_file_location",
                                                 Path(ie_pipeline.details.xml).with_suffix(".mapping"))

    if ir_gen_time_csv_name:
        ir_provider_index = [count for count, step in enumerate(ie_pipeline.steps) if 'ir_provider' in str(step)]
        assert 'model_optimizer_runner' in str(ie_pipeline.steps[ir_provider_index[0]].executor), \
            'IR have been already pregenerated, cannot store ir gen time. Please run pytest without key --pregen_irs'
        assert len(ir_provider_index) == 1, 'Several steps for ir_gen'
        store_data_to_csv(csv_path=os.path.join(Environment.abs_path("mo_out"), ir_gen_time_csv_name),
                          instance=instance, device='CPU', ir_version=ir_version, data_name='ir_gen_time',
                          data=ie_pipeline.steps[ir_provider_index[0]].executor.ir_gen_time,
                          use_mo_legacy_frontend=use_mo_legacy_frontend, use_mo_new_frontend=use_mo_new_frontend,
                          skip_mo_args=skip_mo_args)
    if mem_usage_mo_csv_name:
        ir_provider_index = [count for count, step in enumerate(ie_pipeline.steps) if 'ir_provider' in str(step)]
        if 'model_optimizer_runner' in str(ie_pipeline.steps[ir_provider_index[0]].executor):
            assert len(ir_provider_index) == 1, 'Several steps for ir_gen'
            store_data_to_csv(csv_path=os.path.join(Environment.abs_path("mo_out"), mem_usage_mo_csv_name),
                              instance=instance, device='CPU', ir_version=ir_version, data_name='mem_usage_mo',
                              data=ie_pipeline.steps[ir_provider_index[0]].executor.mem_usage_mo,
                              use_mo_legacy_frontend=use_mo_legacy_frontend, use_mo_new_frontend=use_mo_new_frontend,
                              skip_mo_args=skip_mo_args)
    if load_net_to_plug_time_csv_name or mem_usage_ie_csv_name:
        infer_provider_index = [count for count, step in enumerate(ie_pipeline.steps) if 'infer' in str(step)]
        assert len(infer_provider_index) == 1, 'Several steps for ie_infer'
        if load_net_to_plug_time_csv_name:
            store_data_to_csv(csv_path=os.path.join(Environment.abs_path("mo_out"), load_net_to_plug_time_csv_name),
                              instance=instance, device=ie_pipeline.steps[infer_provider_index[0]].executor.device,
                              ir_version=ir_version, data_name='load_net_to_plug_time',
                              data=ie_pipeline.steps[infer_provider_index[0]].executor.load_net_to_plug_time,
                              use_mo_legacy_frontend=use_mo_legacy_frontend, use_mo_new_frontend=use_mo_new_frontend,
                              skip_mo_args=skip_mo_args)
        if mem_usage_ie_csv_name:
            store_data_to_csv(csv_path=os.path.join(Environment.abs_path("mo_out"), mem_usage_ie_csv_name),
                              instance=instance, device=ie_pipeline.steps[infer_provider_index[0]].executor.device,
                              ir_version=ir_version, data_name='mem_usage_ie',
                              data=ie_pipeline.steps[infer_provider_index[0]].executor.mem_usage_ie,
                              use_mo_legacy_frontend=use_mo_legacy_frontend, use_mo_new_frontend=use_mo_new_frontend,
                              skip_mo_args=skip_mo_args)
            
    if model_framework_name.lower() == 'pytorch' and  instance.precision == 'FP16':
        for name, comparators in instance.comparators.items():
            instance.comparators[name]['a_eps'] = 1e-2
            instance.comparators[name]['r_eps'] = 1e-2

    comparators = ComparatorsContainer(
        config=instance.comparators,
        infer_result=ie_pipeline.fetch_results(),
        reference=ref_pipeline.fetch_results(),
        result_aligner=getattr(instance, 'align_results', None,),
        xml=ie_pipeline.details.xml
    )

    log.info("Running comparators:")
    with log_timestamp('comparators'):
        comparators.apply_postprocessors()
        comparators.apply_all()
    status = comparators.report_statuses()
    assert status, "inferred model results != reference results"


def empty_dirs(env_conf):
    test_config = None
    with open(env_conf, 'r') as fd:
        test_config = yaml.load(fd, Loader=yaml.FullLoader)

    for env_clean_dir_flag, test_cfg_dir_to_clean in [("TT_CLEAN_MO_OUT_DIR", 'mo_out'),
                                                 ("TT_CLEAN_PREGEN_IRS_DIR", 'pregen_irs_path'),
                                                 ("TT_CLEAN_INPUT_MODEL_DIR", 'input_model_dir')]:
        clean_flag = True if os.environ.get(env_clean_dir_flag, 'False') == 'True' else False
        if clean_flag:
            dir_to_clean = test_config.get(test_cfg_dir_to_clean, '')
            if os.path.exists(dir_to_clean):
                log.info(f"Clear {dir_to_clean} dir")
                rmtree(dir_to_clean)


def test_run(instance, pregen_irs, ir_gen_time_csv_name, load_net_to_plug_time_csv_name, mem_usage_mo_csv_name,
             mem_usage_ie_csv_name, record_property, use_mo_legacy_frontend, use_mo_new_frontend, skip_mo_args,
             dynamic_inference, prepare_test_info, skip_ir_generation, copy_input_files, env_conf,
             inference_precision_hint):
    try:
        _test_run(instance, pregen_irs, ir_gen_time_csv_name, load_net_to_plug_time_csv_name, mem_usage_mo_csv_name,
                  mem_usage_ie_csv_name, record_property, use_mo_legacy_frontend, use_mo_new_frontend, skip_mo_args,
                  dynamic_inference, prepare_test_info, skip_ir_generation, copy_input_files, inference_precision_hint)
    except Exception as ex:
        raise Exception(f'{timestamp()}') from ex
    finally:
        empty_dirs(env_conf)
