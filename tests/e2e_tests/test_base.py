# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

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
from e2e_tests.common.logger import get_logger
from e2e_tests.common.parsers import pipeline_cfg_to_string
from e2e_tests.test_utils.test_utils import log_timestamp, read_irs_mapping_file, get_ir_tag, check_mo_precision, \
    set_infer_precision_hint, remove_mo_args_oob, store_data_to_csv, timestamp
from e2e_tests.common.common.pipeline import Pipeline
from e2e_tests.common.comparator.container import ComparatorsContainer

pytest_plugins = ('e2e_tests.common.plugins.e2e_test.conftest',)

log = get_logger(__name__)


def _test_run(instance, pregen_irs, record_property, prepare_test_info, inference_precision_hint):
    """Parameterized test.

    :param instance: test instance
    :param pregen_irs: custom fixture. Provides path to a CSV-formatted file with IRs mapping
    :param record_property: extra property for the calling test
    :param instance: test instance
    """
    # Name of tests group
    prepare_test_info['pytestEntrypoint'] = 'E2E TF Hub: Base'

    ir_version = "v11"

    log.info("Running {test_id} test".format(test_id=instance.test_id))
    instance.prepare_prerequisites()

    log.debug("Test scenario:")
    instance_model_pipeline = instance.model
    model_pipeline = Pipeline(instance_model_pipeline)
    log.debug("Model Pipeline:\n{}".format(pipeline_cfg_to_string(model_pipeline._config)))
    model_pipeline.run()
    model = model_pipeline.fetch_results()

    instance_input_pipeline = instance.input
    input_pipeline = Pipeline(instance_input_pipeline, {'model_obj': model})
    log.debug("Input Pipeline:\n{}".format(pipeline_cfg_to_string(input_pipeline._config)))
    input_pipeline.run()
    feed_dict = input_pipeline.fetch_results()

    instance_ie_pipeline = instance.ie_pipeline
    instance_ref_pipeline = instance.ref_pipeline

    ref_pipeline = Pipeline(instance_ref_pipeline, {'model_obj': model, 'feed_dict': feed_dict})

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
            instance.required_params = {"sequence_length": instance.sequence_length} if type(
                instance.sequence_length) == int else {}
            ir_tag = get_ir_tag(instance.__class__.__name__, ir_version, instance.precision,
                                instance.batch, instance.required_params.get("sequence_length", None))
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
                        log.warning('IR was collected successfully, but OVC log was not saved.')
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

    if instance_ie_pipeline.get('infer'):
        instance_ie_pipeline = set_infer_precision_hint(instance, instance_ie_pipeline, inference_precision_hint)
    ie_pipeline = Pipeline(instance_ie_pipeline, {'model_obj': model, 'feed_dict': feed_dict})
    log.debug("Inference Pipeline:\n{}".format(pipeline_cfg_to_string(ie_pipeline._config)))
    ie_pipeline.run()

    comparators = ComparatorsContainer(
        config=instance.comparators,
        infer_result=ie_pipeline.fetch_results(),
        reference=ref_pipeline.fetch_results(),
        result_aligner=getattr(instance, 'align_results', None, ),
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


def test_run(instance, pregen_irs, record_property, prepare_test_info, copy_input_files, env_conf,
             inference_precision_hint):
    try:
        _test_run(instance, pregen_irs, record_property, prepare_test_info, inference_precision_hint)
    except Exception as ex:
        raise Exception(f'{timestamp()}') from ex
    finally:
        empty_dirs(env_conf)
