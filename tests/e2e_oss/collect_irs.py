"""Runner for collecting IRs for models. Prepares CSV-formatted mapping file for IRs reuse
(e.g. E2E tests, IRs_comparator tests).
To prepare IRs, MO options defined in default IE pipeline is used.
"""

import logging as log
import re
import sys
from pathlib import Path
from shutil import copyfile

from e2e_oss.utils.test_utils import get_ir_tag, read_irs_mapping_file, write_irs_mapping_file, remove_mo_args_oob
from tests.utils.e2e.common.pipeline import Pipeline

pytest_plugins = ('e2e_oss.plugins.e2e_test.conftest',)


def _add_ir_to_html_report(xml, record_property):
    if xml is not None:
        ir_http_path = "http://{}".format(re.sub(r"^[/|\\]+", "", str(xml)).replace("\\", "/"))
        record_property("ir_link", ir_http_path)


def test_collect_irs(instance, pregen_irs, record_property, use_mo_legacy_frontend, use_mo_new_frontend, skip_mo_args,
                     prepare_test_info, copy_input_files):
    """ Collect IRs for models and prepare CSV-formatted mapping file for simple navigating and IRs reuse.
    :param instance: test instance
    :param pregen_irs: custom fixture. Provides path to a CSV-formatted file with IRs mapping
    """
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.DEBUG, stream=sys.stdout)

    # Name of tests group
    prepare_test_info['pytestEntrypoint'] = 'Collect IRs'

    ir_version = "v11"

    assert pregen_irs, "Specify --pregen_irs to generate file with IRs mapping"

    irs_mapping_path = pregen_irs
    irs_mapping = {}
    if irs_mapping_path.exists():
        irs_mapping = read_irs_mapping_file(irs_mapping_path, lock_access=True)
    instance.required_params = {"sequence_length": instance.sequence_length} if type(instance.sequence_length) == int else {}
    ir_tag = get_ir_tag(instance.__class__.__name__, ir_version, instance.precision,
                        instance.batch, instance.required_params.get("sequence_length", None), use_mo_legacy_frontend,
                        use_mo_new_frontend, skip_mo_args)
    if ir_tag in irs_mapping:
        log.info("Record is already in mapping file: {}. Skipping test ...".format({ir_tag: irs_mapping[ir_tag]}))
        _add_ir_to_html_report(irs_mapping[ir_tag][2], record_property)
        return

    # Step that prepares some fields which aren't defined in a pipelines by default
    instance.prepare_prerequisites()

    if "get_ir" not in instance.ie_pipeline:
        log.info("No step for IR generation, skipping test ...")
        return

    if "mo" not in instance.ie_pipeline["get_ir"].keys():
        if "pregenerated" in instance.ie_pipeline["get_ir"]:
            log.info("Reuse already pre-generated IR. Copying files ...")
            irs_mapping[ir_tag] = [True, None]
            for field in ["xml", "bin"]:
                # TODO: handle MO mapping
                src_path = Path(instance.ie_pipeline["get_ir"]['pregenerated'][field])
                dst_path = Path(instance.environment['pregen_irs_path']) / Path(ir_tag).with_suffix(src_path.suffix)
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                copyfile(src_path, dst_path)
                irs_mapping[ir_tag].append(dst_path)
            write_irs_mapping_file(irs_mapping_path, ir_tag, *irs_mapping[ir_tag])
            _add_ir_to_html_report(irs_mapping[ir_tag][2], record_property)
            return
        # TODO: add support of "omz_model_downloader' key
        log.info("No proper step to pre-generate IR. Skipping test ...")
        return

    if instance.batch > 1 and "network_modifiers" in instance.ie_pipeline.get("infer", {}).get("ie_sync", {}).keys():
        # Try to skip generating IR with batch == 1 twice. Test with batch > 1 will
        # reuse IR with batch == 1 and change batch via IE
        ref_ir_tag = get_ir_tag(instance.__class__.__name__, ir_version, instance.precision,
                                batch=1, sequence_length=instance.required_params.get("sequence_length", None),
                                use_mo_legacy_frontend=use_mo_legacy_frontend, use_mo_new_frontend=use_mo_new_frontend,
                                skip_mo_args=skip_mo_args)
        log.info("Trying to reuse IR with tag '{}'".format(ref_ir_tag))
        if ref_ir_tag in irs_mapping:
            write_irs_mapping_file(irs_mapping_path, ir_tag, *irs_mapping[ref_ir_tag])
            _add_ir_to_html_report(irs_mapping[ref_ir_tag][2], record_property)
            return
        log.info("Tag '{}' not found in mapping file".format(ref_ir_tag))

    if instance.ie_pipeline.get('load_pytorch_model'):
        get_ir_pipeline = {
            "load_pytorch_model": instance.ie_pipeline["load_pytorch_model"],
            "get_ir": instance.ie_pipeline['get_ir']
        }
    else:
        get_ir_pipeline = {"get_ir": instance.ie_pipeline["get_ir"]}
    get_ir_pipeline["get_ir"]["mo"].update({"mo_out": instance.environment["pregen_irs_path"],
                                            "target_ir_name": ir_tag,
                                            "use_mo_cmd_tool": instance.use_mo_cmd_tool})

    if use_mo_legacy_frontend:
        instance.ie_pipeline["get_ir"]["mo"]["additional_args"].update({"use_legacy_frontend": True})

    if use_mo_new_frontend:
        instance.ie_pipeline["get_ir"]["mo"]["additional_args"].update({"use_new_frontend": True})

    if skip_mo_args:
        mo_additional_args = get_ir_pipeline['get_ir']['mo'].get('additional_args', {})
        if mo_additional_args:
            get_ir_pipeline['get_ir']['mo']['additional_args'] = remove_mo_args_oob(skip_mo_args,
                                                                                    mo_additional_args,
                                                                                    get_ir_pipeline)

    exe_pipeline = Pipeline(get_ir_pipeline)
    try:
        # WA for PyTorch pretrained and Torchvision models
        # ONNX model representation dumped in ref_collection provider initialization
        # So we have to init the pipeline to create .onnx model to feed to MO
        # For other frameworks it will be extra step with no effect
        Pipeline(getattr(instance, "ref_pipeline", {}))

        log.info("Generating IR ...")
        exe_pipeline.run()
    finally:
        mo_log = exe_pipeline.details.mo_log
        xml = exe_pipeline.details.xml
        bin = Path(xml).with_suffix(".bin") if xml is not None else None
        status = Path(xml).exists() if xml is not None else False
        write_irs_mapping_file(irs_mapping_path, ir_tag, status, mo_log, xml, bin)
        _add_ir_to_html_report(xml, record_property)
