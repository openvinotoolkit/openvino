"""Main entry-point to run E2E OSS tests.

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
from pathlib import Path
from collections import OrderedDict
# pylint:disable=invalid-name
from copy import deepcopy
from math import ceil

from e2e_oss.utils.test_utils import check_mo_precision, read_irs_mapping_file, get_ir_tag
from e2e_oss.common_utils.parsers import pipeline_cfg_to_string
from e2e_oss.pipelines.pipeline_templates.comparators_template import eltwise_comparators
from utils.e2e.common.pipeline import Pipeline
from utils.e2e.comparator.container import ComparatorsContainer

pytest_plugins = ('e2e_oss.plugins.e2e_test.conftest',)

log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.DEBUG, stream=sys.stdout)


def pytest_generate_tests(metafunc):
    setattr(metafunc, 'test_add_args_to_parametrize', ['cpu_streams', 'gpu_streams'])


def share_ir_path(source_pipeline, dest_pipeline, source_pipeline_step_name="infer", dest_pipeline_step_name="infer"):
    """
    Copy .xml and .bin between pipelines
    :param source_pipeline: pipeline where to get .xml and .bin file paths
    :param dest_pipeline: pipeline where to set .xml and .bin file paths
    :param source_pipeline_step_name: name of a step in source pipeline from which to get xml and bin attributes
    :param dest_pipeline_step_name: name of a step in destination pipeline where xml and bin attributes have to be set
    :return: updated pipeline
    """
    for i, dest_step in enumerate(dest_pipeline.steps):
        if dest_step.__step_name__ == dest_pipeline_step_name:
            for source_step in source_pipeline.steps:
                if source_step.__step_name__ == source_pipeline_step_name:
                    dest_pipeline.steps[i].executor.xml = source_step.executor.xml
                    dest_pipeline.steps[i].executor.bin = source_step.executor.bin
    return dest_pipeline


def test_compare_requests(instance, gpu_throughput_mode, cpu_throughput_mode, prepare_test_info):
    """Parameterized test.
    :param instance: test instance
    """
    log.info("\033[95m" + "Running {test_name} test".format(test_name=instance.__class__.__name__) + "\033[0m")

    # Name of tests group
    prepare_test_info['pytestEntrypoint'] = 'Compare requests'

    instance.prepare_prerequisites()
    instance_ie_pipeline = instance.ie_pipeline
    check_mo_precision(instance_ie_pipeline)

    infer_step = instance_ie_pipeline.get("infer")
    nireq = 2
    if "ie_sync" in infer_step:
        instance_ie_pipeline.pop("postprocess", None)
        ref_pipeline = Pipeline(instance_ie_pipeline)
        log.debug("Reference Pipeline:\n{}".format(pipeline_cfg_to_string(ref_pipeline._config)))
        infer_step["ie_sync"].update({"num_requests": nireq})
        instance_ie_pipeline['infer'] = {"ie_async": infer_step["ie_sync"]}

        # To avoid running MO twice we are getting xml and bin files paths from reference run
        instance_ie_pipeline.pop("get_ir")
        if gpu_throughput_mode and instance.gpu_streams is not None:
            nireq = 2 if instance.gpu_streams == "GPU_THROUGHPUT_AUTO" else int(instance.gpu_streams) * 2
            cfg_update = {"plugin_config": {"GPU_THROUGHPUT_STREAMS": str(instance.gpu_streams)},
                          "plugin_cfg_target_device": "GPU",
                          "num_requests": nireq}
            instance_ie_pipeline['infer']["ie_async"].update(cfg_update)
        if cpu_throughput_mode and instance.cpu_streams is not None:
            nireq = 2 if instance.cpu_streams == "CPU_THROUGHPUT_AUTO" else int(instance.cpu_streams) * 2
            cfg_update = {"plugin_config": {"CPU_THROUGHPUT_STREAMS": str(instance.cpu_streams)},
                          "plugin_cfg_target_device": "CPU",
                          "num_requests": nireq}
            instance_ie_pipeline['infer']["ie_async"].update(cfg_update)
        ie_pipeline = Pipeline(instance_ie_pipeline)
        log.debug("Inference Pipeline:\n{}".format(pipeline_cfg_to_string(ie_pipeline._config)))
    elif not infer_step:
        raise AttributeError("Infer step in ie_pipeline is absent or empty!")
    else:
        raise AttributeError("Incompatible action provider '{}' for 'infer' step! "
                             "Expected 'ie_sync'".format(list(infer_step.keys())[0]))

    log.info("\033[95m" + "Running reference pipeline:" + "\033[0m")
    ref_pipeline.run()

    # To avoid running MO twice we are getting xml and bin files paths from reference run
    ie_pipeline = share_ir_path(source_pipeline=ref_pipeline, dest_pipeline=ie_pipeline)
    log.info("\033[95m" + "Running inference pipeline:" + "\033[0m")
    ie_pipeline.run()

    statuses = []
    for i in range(nireq):
        log.info("\033[95m" + "Comparing {} infer request with sync inference run ...".format(i + 1) + "\033[0m")
        comparators = ComparatorsContainer(
            config=eltwise_comparators(precision=instance.precision, device=instance.device),
            infer_result=ref_pipeline.fetch_results(),
            reference=ie_pipeline.fetch_results()[i],
            result_aligner=getattr(instance, 'align_results', None))
        comparators.apply_postprocessors()
        comparators.apply_all()
        if comparators.report_statuses():
            log.info("\033[92m" + "Comparison passed " + u'\u2714' + "\033[0m")
        else:
            log.info("\033[91m" + "Comparison failed " + u'\u274c' + "\033[0m")
        statuses.append(comparators.report_statuses())
    assert all(statuses), "inferred model results != reference results"


_transformers = [{},
                 {"add_gaussian_noise": {"mean": 0.1, "sigma": 0.2}},
                 {"add_gaussian_noise": {"mean": 0.2, "sigma": 0.1}},
                 {"add_gaussian_noise": {"mean": 0., "sigma": 0.4}},
                 ]


def test_compare_with_refs_multi_image(instance, gpu_throughput_mode, cpu_throughput_mode,
                                       pregen_irs, record_property, prepare_test_info):
    log.info("\033[95m" + "Running {test_name} test".format(test_name=instance.__class__.__name__) + "\033[0m")

    # Name of tests group
    prepare_test_info['pytestEntrypoint'] = 'Compare with reference multi image'

    instance.prepare_prerequisites()
    instance_ie_pipeline = instance.ie_pipeline
    instance_ref_pipeline = instance.ref_pipeline

    # Target MULTI plugin configuration going to contain 2 devices,
    # so nireq=4 intended to feed two request to each device
    nireq = 4

    if gpu_throughput_mode and instance.gpu_streams is not None:
        nireq = 2 if instance.gpu_streams == "GPU_THROUGHPUT_AUTO" else int(instance.gpu_streams) * 2
    if cpu_throughput_mode and instance.cpu_streams is not None:
        nireq = 2 if instance.cpu_streams == "CPU_THROUGHPUT_AUTO" else int(instance.cpu_streams) * 2
    transformers = _transformers * ceil(nireq / len(_transformers))
    inputs = []
    ref_results = []

    for i in range(nireq):
        if hasattr(instance, "ref_collection"):
            ref_pipeline_cfg = deepcopy(instance.ref_collection['pipeline'])
        else:
            ref_pipeline_cfg = deepcopy(instance_ref_pipeline)

        transformer = transformers[i]
        prepare_input_pipeline_cfg = OrderedDict([("read_input", ref_pipeline_cfg["read_input"])])
        if transformer:
            prepare_input_pipeline_cfg["preprocess"] = transformer
            prepare_input_pipeline_cfg.move_to_end("preprocess", last=True)

        prepare_input_pipeline = Pipeline(prepare_input_pipeline_cfg)
        log.info("\033[34m" + "Running input preparation pipeline:" + "\033[0m")
        log.debug("Input Preparation Pipeline:\n{}".format(pipeline_cfg_to_string(prepare_input_pipeline._config)))
        prepare_input_pipeline.run()
        req_input = prepare_input_pipeline.fetch_results()
        inputs.append(deepcopy(req_input))

        ref_pipeline_cfg["read_input"] = OrderedDict([("external_data", {"data": req_input})])
        ref_pipeline_cfg["postprocess"] = instance_ref_pipeline.get("postprocess", {})
        ref_pipeline_cfg.move_to_end("postprocess", last=True)
        ref_pipeline = Pipeline(ref_pipeline_cfg)
        log.info("\033[34m" + "Running reference pipeline:" + "\033[0m")
        log.debug("Reference Pipeline:\n{}".format(pipeline_cfg_to_string(ref_pipeline._config)))
        ref_pipeline.run()
        ref_results.append(ref_pipeline.fetch_results())

    infer_step = instance_ie_pipeline.get("infer")

    ir_version = "v11"
    if pregen_irs and "get_ir" in instance_ie_pipeline:
        try:
            log.info("Searching pre-generated IR in IR's mapping: {} ...".format(pregen_irs))
            irs_mapping = read_irs_mapping_file(pregen_irs)
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
                        ir_http_path = "http://{}".format(re.sub(r"^[/|\\]+", "", xml).replace("\\", "/"))
                        record_property("ir_link", ir_http_path)
                        instance_ie_pipeline["get_ir"] = {"pregenerated": {"xml": xml, "bin": bin}}
        except Exception as e:
            log.error("Search of pre-generated IR failed with error: {err}. IR will be generated in runtime ..."
                      .format(err=e))

    if "ie_sync" in infer_step:
        instance_ie_pipeline["read_input"] = {"external_data": {"data": inputs}}
        infer_step["ie_sync"].update({"num_requests": nireq, "multi_image": True})
        instance_ie_pipeline['infer'] = {"ie_async": infer_step["ie_sync"]}
        if gpu_throughput_mode and instance.gpu_streams is not None:
            cfg_update = {"plugin_config": {"GPU_THROUGHPUT_STREAMS": str(instance.gpu_streams)},
                          "plugin_cfg_target_device": "GPU",
                          "num_requests": nireq}
            instance_ie_pipeline['infer']["ie_async"].update(cfg_update)
        if cpu_throughput_mode and instance.cpu_streams is not None:
            cfg_update = {"plugin_config": {"CPU_THROUGHPUT_STREAMS": str(instance.cpu_streams)},
                          "plugin_cfg_target_device": "CPU",
                          "num_requests": nireq}
            instance_ie_pipeline['infer']["ie_async"].update(cfg_update)

        # Preprocessors and postprocessors have to be run in separate pipelines
        # for each input image and output results
        ie_pipeline_prepocess_cfg = instance_ie_pipeline.pop("preprocess", {})
        ie_pipeline_postproc_cfg = instance_ie_pipeline.pop("postprocess", {})

        for i, img in enumerate(inputs):
            preprocess_pipeline_cfg = OrderedDict([("read_input", {"external_data": {"data": img}}),
                                                   ("preprocess", ie_pipeline_prepocess_cfg)])
            preprocess_pipeline = Pipeline(preprocess_pipeline_cfg)
            log.info("\033[34m" + "Running preprocessing pipeline for image {}:".format(i + 1) + "\033[0m")
            log.debug("Preprocessing Pipeline:\n{}".format(pipeline_cfg_to_string(preprocess_pipeline._config)))
            preprocess_pipeline.run()
            inputs[i] = preprocess_pipeline.fetch_results()

    elif not infer_step:
        raise AttributeError("\033[91m" + "Infer step in ie_pipeline is absent or empty!" + u'\u274c' + "\033[0m")
    else:
        raise AttributeError("\033[91m" + "Incompatible action provider '{}' for 'infer' step! "
                                          "Expected 'ie_sync'".format(list(infer_step.keys())[0])
                             + u'\u274c' + "\033[0m")

    ie_pipeline = Pipeline(instance_ie_pipeline)
    log.info("\033[34m" + "Running inference pipeline:" + "\033[0m")
    log.debug("Inference Pipeline:\n{}".format(pipeline_cfg_to_string(ie_pipeline._config)))
    ie_pipeline.run()
    ie_results = ie_pipeline.fetch_results()

    for i, data in enumerate(ie_results):
        postrpoc_pipeline_cfg = OrderedDict([("read_input", {"external_data": {"data": data}}),
                                             ("postprocess", ie_pipeline_postproc_cfg)])
        postproc_pipeline = Pipeline(postrpoc_pipeline_cfg)
        log.info("\033[34m" + "Running postprocess pipeline for image {}:".format(i + 1) + "\033[0m")
        log.debug("Postprocess Pipeline:\n{}".format(pipeline_cfg_to_string(postproc_pipeline._config)))
        postproc_pipeline.run()
        ie_results[i] = postproc_pipeline.fetch_results()

    statuses = []
    for i in range(nireq):
        comparators = ComparatorsContainer(
            config=instance.comparators,
            infer_result=ie_results[i],
            reference=ref_results[i],
            result_aligner=getattr(instance, 'align_results', None))
        log.info("\033[34m" + "Comparing {} infer request with reference ...".format(i + 1) + "\033[0m")
        comparators.apply_postprocessors()
        comparators.apply_all()
        if comparators.report_statuses():
            log.info("\033[92m" + "Comparison passed " + u'\u2714' + "\033[0m")
        else:
            log.info("\033[91m" + "Comparison failed " + u'\u274c' + "\033[0m")
        statuses.append(comparators.report_statuses())

    assert all(statuses), "inferred model results != reference results"
