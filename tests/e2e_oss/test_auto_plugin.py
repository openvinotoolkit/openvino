"""Entry-point to run E2E OSS tests with AUTO PLUGIN.
This entry-point is used to compare two pipelines: with <plugin> and AUTO:<plugin>
where pipeline with <plugin> is reference pipeline

Default run:
$ pytest test.py

Options[*]:
--modules       Paths to tests
--env_conf      Path to environment config
--test_conf     Path to test config

[*] For more information see conftest.py
"""
import re
from copy import deepcopy
from pathlib import Path

# pylint:disable=invalid-name

from common_utils.logger import get_logger
from tests.e2e_oss._utils.test_utils import set_infer_precision_hint, read_irs_mapping_file, get_ir_tag, \
    check_mo_precision
from tests.e2e_oss.common_utils.test_utils import name_aligner
from tests.e2e_oss.test import empty_dirs
from tests.utils.e2e.common.pipeline import Pipeline
from tests.utils.e2e.comparator.container import ComparatorsContainer

pytest_plugins = ('e2e_oss.plugins.e2e_test.conftest',)

log = get_logger(__name__)


def _test_run(instance, pregen_irs, record_property, prepare_test_info, copy_input_files, inference_precision_hint,
              use_mo_cmd_tool):
    """Parameterized test.

    :param instance: test instance
    :param pregen_irs: custom fixture. Provides path to a CSV-formatted file with IRs mapping
    :param record_property: extra property for the calling test
    :param instance: test instance
    """

    # Name of tests group
    prepare_test_info['pytestEntrypoint'] = 'E2E: AUTO Plugin'

    ir_version = "v11"

    log.info("Running {test_id} test".format(test_id=instance.test_id))
    instance.prepare_prerequisites()
    device = instance.device

    instance_ie_pipeline = instance.ie_pipeline
    instance_ref_pipeline = deepcopy(instance_ie_pipeline)

    instance_ie_pipeline['infer'][next(iter(instance_ie_pipeline['infer']))]['device'] = 'AUTO:{}'.format(device)

    for pipeline in [instance_ie_pipeline, instance_ref_pipeline]:
        pipeline = set_infer_precision_hint(instance, pipeline, inference_precision_hint)

        if pregen_irs and "get_ir" in pipeline:
            try:
                log.info("Searching pre-generated IR in IR's mapping: {} ...".format(pregen_irs))
                irs_mapping = read_irs_mapping_file(pregen_irs)
                instance.required_params = {"sequence_length": instance.sequence_length} if type(instance.sequence_length) == int else {}
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
                            pipeline["get_ir"] = {"pregenerated": {"xml": xml, "bin": bin}}
            except Exception as ex:
                log.error("Search of pre-generated IR failed with error: {err}."
                          " IR will be generated in runtime ...".format(err=ex))

    ref_pipeline = Pipeline(instance_ref_pipeline)

    check_mo_precision(instance_ie_pipeline)

    log.info("Running reference pipeline:")
    ref_pipeline.run()

    if use_mo_cmd_tool:
        xml_file = Path(ref_pipeline.details.xml)
        bin_file = xml_file.with_suffix('.bin')

        if 'pregenerated' not in instance_ie_pipeline['get_ir']:
            instance_ie_pipeline['get_ir'] = {"pregenerated": {"xml": xml_file, "bin": bin_file}}
            log.info(f"IR from previous pipeline will be used, path to folder with IR: \n{xml_file.parent}")

    log.info("Running inference pipeline with AUTO Plugin")
    ie_pipeline = Pipeline(instance_ie_pipeline)
    ie_pipeline.run()

    comparators = ComparatorsContainer(
        config=instance.comparators,
        infer_result=ie_pipeline.fetch_results(),
        reference=ref_pipeline.fetch_results(),
        result_aligner=name_aligner)

    log.info("Running comparators:")
    comparators.apply_postprocessors()
    comparators.apply_all()
    status = comparators.report_statuses()
    assert status, "inferred model results != reference results"


def test_run(instance, pregen_irs, record_property, prepare_test_info, copy_input_files, env_conf,
             inference_precision_hint, use_mo_cmd_tool):
    try:
        _test_run(instance, pregen_irs, record_property, prepare_test_info, copy_input_files, inference_precision_hint,
                  use_mo_cmd_tool)
    except Exception as ex:
        raise ex
    finally:
        empty_dirs(env_conf)
