# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Basic high-level plugin file for pytest.

See [Writing plugins](https://docs.pytest.org/en/latest/writing_plugins.html)
for more information.

This plugin adds the following command-line options:

* `--modules` - Paths to modules to be run by pytest (these can contain tests,
  references, etc.). Format: Unix style pathname patterns or .py files.
* `--env_conf` - Path to environment configuration file. Used to initialize test
  environment. Format: yaml file.
* `--test_conf` - Path to test configuration file. Used to parameterize tests.
  Format: yaml file.
* `--dry_run` - Specifies that reference collection should not store collected
  results to filesystem.
* `--bitstream` - Path to bitstream to ran tests with.
* `--tf_models_version` - TensorFlow models version.
"""
import json
import logging as log
import os
import platform
import re
import time
from contextlib import contextmanager
from inspect import getsourcefile
from pathlib import Path
import shutil

# pylint:disable=import-error
import pytest
from jsonschema import validate, ValidationError

from e2e_tests.test_utils.test_utils import get_framework_from_model_ex
from e2e_tests.test_utils.env_tools import Environment


@contextmanager
def import_from(path):
    """ Set import preference to path"""
    os.sys.path.insert(0, os.path.realpath(path))
    yield
    os.sys.path.remove(os.path.realpath(path))


def pytest_addoption(parser):
    """Specify command-line options for all plugins"""
    if getattr(parser, "after_preparse", False):
        return
    parser.addoption(
        "--modules",
        nargs='+',
        help="Path to test modules",
        default=["pipelines"]
    )
    parser.addoption(
        "--env_conf",
        action="store",
        help="Path to environment configuration file",
        default="env_config_local.yml"
    )
    parser.addoption(
        "--test_conf",
        action="store",
        help="Path to test configuration file",
        default="test_config_local.yml"
    )
    parser.addoption(
        "--dry_run",
        action="store_true",
        help="Dry run reference collection: not saving to filesystem",
        default=False
    )
    parser.addoption(
        "--collect_output",
        action="store",
        help="Path to dry run output file",
        default=None
    )
    parser.addoption(
        "--base_rules_conf",
        action="store",
        help="Path to base test rules configuration file",
        default="base_test_rules.yml"
    )
    parser.addoption(
        "--reshape_rules_conf",
        action="store",
        help="Path to reshape test rules configuration file",
        default="reshape_test_rules.yml"
    )
    parser.addoption(
        "--dynamism_rules_conf",
        action="store",
        help="Path to dynamism test rules configuration file",
        default="dynamism_test_rules.yml"
    )
    parser.addoption(
        "--bitstream",
        action="store",
        help="Bitstream path; run tests for models supported by this bitstream",
        default=""
    )
    parser.addoption(
        "--pregen_irs",
        type=Path,
        help="Name of IR's mapping file (CSV-formatted) to use pre-generated IRs in tests."
             " File and pre-generated IRs will be located in `pregen_irs_path` defined in environment config",
        default=None
    )
    parser.addoption(
        "--ir_gen_time_csv_name",
        action="store",
        help="Name for csv file with IR generation time",
        default=False
    )
    parser.addoption(
        "--load_net_to_plug_time_csv_name",
        action="store",
        help="Name for csv file with load net to plugin time",
        default=False
    )
    parser.addoption(
        "--mem_usage_mo_csv_name",
        action="store",
        help="Name for csv file with MO memory usage information",
        default=False
    )
    parser.addoption(
        "--mem_usage_ie_csv_name",
        action="store",
        help="Name for csv file with IE memory usage information",
        default=False
    )
    parser.addoption(
        "--gpu_throughput_mode",
        action="store_true",
        help="Enable GPU_THROUGHPUT_STREAMS mode for multi_request tests",
        default=False
    )
    parser.addoption(
        "--cpu_throughput_mode",
        action="store_true",
        help="Enable GPU_THROUGHPUT_STREAMS mode for multi_request tests",
        default=False
    )
    parser.addoption(
        "--tf_models_version",
        action="store",
        help="Specify TensorFlow models version",
        default=None
    )
    parser.addoption(
        "--dynamism_type",
        action="store",
        help="This option is used in dynamism tests. Possible types: negative_ones, range_values",
        default=None
    )
    parser.addoption(
        "--skip_mo_args",
        help="List of args to remove from MO command line",
        required=False
    )
    parser.addoption(
        "--dynamic_inference",
        help="Enable dynamic inference mode",
        action="store_true",
        default=False
    )
    parser.addoption(
        "--db_url",
        type=str,
        help="Url to send post request to DataBase. http://<Server_name>/api/v1/e2e/push-2-db-facade",
        action="store",
        default=None
    )
    parser.addoption(
        '--infer_binary_path',
        type=Path,
        help='Path to timetest_infer/memtest_infer binary file',
        default=None
    )
    parser.addoption(
        "--consecutive_infer",
        action="store_true",
        help="This option is used in dynamism tests. Specify if values from input_descriptor should be used",
        default=False
    )
    parser.addoption(
        "--skip_ir_generation",
        action="store_true",
        help="Load model to IE plugin as is (uses ONNX or PDPD Importer)",
        default=False
    )
    parser.addoption(
        '--inference_precision_hint',
        help='Inference Precision hint for device',
        required=False
    )
    parser.addoption(
        "--convert_pytorch_to_onnx",
        action="store_true",
        help="Whether or not use pytorch to onnx OMZ converter",
        default=False
    )


@pytest.fixture(scope="session")
def modules(request):
    """Fixture function for command-line option."""
    return request.config.getoption('modules')


@pytest.fixture(scope="session")
def env_conf(request):
    """Fixture function for command-line option."""
    return request.config.getoption('env_conf')


@pytest.fixture(scope="session")
def test_conf(request):
    """Fixture function for command-line option."""
    return request.config.getoption('test_conf')


@pytest.fixture(scope="session")
def dry_run(request):
    """Fixture function for command-line option."""
    return request.config.getoption('dry_run')


@pytest.fixture(scope="session")
def base_rules_conf(request):
    """Fixture function for command-line option."""
    return request.config.getoption('base_rules_conf')


@pytest.fixture(scope="session")
def dynamism_rules_conf(request):
    """Fixture function for command-line option."""
    return request.config.getoption('dynamism_rules_conf')


@pytest.fixture(scope="session")
def reshape_rules_conf(request):
    """Fixture function for command-line option."""
    return request.config.getoption('reshape_rules_conf')


@pytest.fixture(scope="session")
def bitstream(request):
    """Fixture function for command-line option."""
    return request.config.getoption('bitstream')


@pytest.fixture(scope="session")
def pregen_irs(request):
    """Fixture function for command-line option."""
    path = request.config.getoption('pregen_irs')
    if path:
        # Create sub-folders and file before tests to make execution via pytest-xdist safer
        path = Path(Environment.env['pregen_irs_path']) / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
    return path


@pytest.fixture(scope="session")
def ir_gen_time_csv_name(request):
    """Fixture function for command-line option."""
    return request.config.getoption('ir_gen_time_csv_name')


@pytest.fixture(scope="session")
def load_net_to_plug_time_csv_name(request):
    """Fixture function for command-line option."""
    return request.config.getoption('load_net_to_plug_time_csv_name')


@pytest.fixture(scope="session")
def mem_usage_mo_csv_name(request):
    """Fixture function for command-line option."""
    return request.config.getoption('mem_usage_mo_csv_name')


@pytest.fixture(scope="session")
def mem_usage_ie_csv_name(request):
    """Fixture function for command-line option."""
    return request.config.getoption('mem_usage_ie_csv_name')


@pytest.fixture(scope="session")
def gpu_throughput_mode(request):
    """Fixture function for command-line option."""
    if request.config.getoption('gpu_throughput_mode') and request.config.getoption('cpu_throughput_mode'):
        raise ValueError("gpu_throughput_mode and cpu_throughput_mode options can't be specified simultaneously")
    return request.config.getoption('gpu_throughput_mode')


@pytest.fixture(scope="session")
def cpu_throughput_mode(request):
    """Fixture function for command-line option."""
    if request.config.getoption('gpu_throughput_mode') and request.config.getoption('cpu_throughput_mode'):
        raise ValueError("gpu_throughput_mode and cpu_throughput_mode options can't be specified simultaneously")
    return request.config.getoption('cpu_throughput_mode')


@pytest.fixture(scope="session")
def dynamism_type(request):
    """Fixture function for command-line option."""
    return request.config.getoption('dynamism_type')


@pytest.fixture(scope="session")
def infer_binary_path(request):
    """Fixture function for command-line option."""
    return request.config.getoption('infer_binary_path')


@pytest.fixture(scope="session")
def skip_mo_args(request):
    """Fixture function for command-line option."""
    return request.config.getoption('skip_mo_args')


@pytest.fixture(scope="session")
def dynamic_inference(request):
    """Fixture function for command-line option."""
    return request.config.getoption('dynamic_inference')


@pytest.fixture(scope="session")
def consecutive_infer(request):
    """Fixture function for command-line option."""
    return request.config.getoption('consecutive_infer')


@pytest.fixture(scope="session")
def skip_ir_generation(request):
    """Fixture function for command-line option."""
    return request.config.getoption('skip_ir_generation')


@pytest.fixture(scope="session")
def inference_precision_hint(request):
    """Fixture function for command-line option."""
    return request.config.getoption('inference_precision_hint')


@pytest.fixture(scope="session")
def convert_pytorch_to_onnx(request):
    """Fixture function for command-line option."""
    return request.config.getoption('convert_pytorch_to_onnx')


def pytest_collection_finish(session):
    """ Pytest hook for test collection.
    Dump list of tests to 'dry_run.csv' file.
    :param session: session object
    :return: None
    """
    if session.config.getoption('collect_output'):
        with import_from(getsourcefile(lambda: 0) + '/../../../../common'):
            from metrics_utils import write_csv

        collect_output = session.config.getoption('collect_output') or \
                         Environment.abs_path('logs_dir', 'dry_run.csv')
        collect_only = session.config.getoption('collectonly')
        for item in session.items:
            # Extract test name
            match = re.search(r'\[((\w|\.|-)+)\]', item.name)

            if match and match.group(1):
                test_name = match.group(1)
                # Write csv
                if collect_only:
                    session_items_params = item._fixtureinfo.name2fixturedefs.get('instance')[0].params
                    write_csv({'test_filter': test_name}, collect_output, ',')
            else:
                log.error('Unable to extract test name from string "{}"'.format(item.name))


@pytest.fixture(scope="function")
def prepare_test_info(request, instance):
    """
    Fixture for preparing and validating data to submit to a database.
    """
    setattr(request.node._request, 'test_info', {})

    test_id = getattr(instance, 'test_id')
    network_name = test_id

    # add test info
    info = {
        # results will be added immediately before uploading to DB in `pytest_runtest_makereport`
        'insertTime': 0,  # Current date when call upload to DataBase
        'topLevelLink': '',
        'lowLevelLink': os.getenv('RUN_DISPLAY_URL', 'Local run'),
        'subset': os.getenv('model_type', 'Not set or precommit'),
        'platform': os.getenv('node_selector', 'Undefined'),
        'os': os.getenv('os', 'Undefined'),
        'framework': '',
        'network': network_name,
        'inputsize': '',
        'dynamismType': '',
        # TODO: remove 'fusing' key, when this will dropped in DataBase
        'fusing': False,
        'device': getattr(instance, 'device'),
        'precision': getattr(instance, 'precision'),
        'model': '',
        'result': '',
        'duration': 0,
        'links': '',
        'log': '',
        'moTime': 0,
        'moMemory': 0,
        'links2JiraTickets': [],
        'pytestEntrypoint': '',
        'ext': ''
    }
    request.node._request.test_info.update(info)

    yield request.node._request.test_info
    if not request.config.getoption('db_url'):
        return

    request.node._request.test_info.update({
        'insertTime': time.time(),
        'topLevelLink': get_ie_version(),
        'moTime': get_mo_time(request.node._request.test_info['log']),
        'moMemory': get_mo_memory(request.node._request.test_info['log']),
        'model': get_model_path(request.node._request.test_info['log'])
    })
    request.node._request.test_info.update({
        'framework': get_framework_from_model_ex(instance.definition_path)
    })
    # TODO: remove 'fusing' key, when this will dropped in DataBase
    schema = """
        {
            "type": "object",
            "properties": {
                "insertTime": {"type": "number"},
                "topLevelLink": {"type": "string"},
                "lowLevelLink": {"type": "string"},
                "subset": {"type": "string"},
                "platform": {"type": "string"},
                "os": {"type": "string"},
                "framework": {"type": "string"},
                "network": {"type": "string"},
                "batch": {"type": "integer"},
                "device": {"type": "string"},
                "fusing": {"type": "boolean"},
                "precision": {"type": "string"},
                "result": {"type": "string"},
                "duration": {"type": "number"},
                "links": {"type": "string"},
                "log": {"type": "string"},
                "model": {"type": "string"},
                "moTime": {"type": "number"},
                "moMemory": {"type": "number"},
                "links2JiraTickets": {"type": "array"},
                "pytestEntrypoint": {"type": "string"},
                "ext": {"type": "string"}
            },
            "required": ["insertTime", "topLevelLink", "lowLevelLink", "subset", "platform",
                         "os", "framework", "network", "batch", "device", "precision",
                         "result", "duration", "links", "log", "model", "moTime", "moMemory",
                         "links2JiraTickets", "pytestEntrypoint", "ext" ],
            "additionalProperties": true
        }
        """

    schema = json.loads(schema)
    try:
        validate(instance=request.node._request.test_info, schema=schema)
    except ValidationError:
        raise

    upload_db(data=request.node._request.test_info, url=request.config.getoption('db_url'))


def upload_db(data, url):
    from requests import post
    from requests.structures import CaseInsensitiveDict

    headers = CaseInsensitiveDict()
    headers["accept"] = "application/json"
    headers["Content-Type"] = "application/json"

    resp = post(url, headers=headers, data=json.dumps({'data': [data]}))

    if resp.status_code == 200:
        log.info(f'Data successfully uploaded to DB: {url}')
    else:
        log.error(f'Upload data failed. DB return: code - {resp.status_code}\n'
                  f'Message - {resp.text}')


def get_ie_version():
    import openvino.runtime as rt
    version = rt.get_version()
    return version if version else "Not_found"


def get_mo_time(test_log):
    pattern_time = r'Total execution time:\s*(\d+\.?\d*)\s*seconds'
    mo_time = re.search(pattern_time, test_log)
    return float(mo_time.group(1)) if mo_time else 0


def get_mo_memory(test_log):
    pattern_memory = r'Memory consumed:\s*(\d+)\s*MB.'
    memory = re.search(pattern_memory, test_log)
    return float(memory.group(1)) if memory else 0


def get_model_path(test_log):
    pattern_path = r'Input model was copied from \s*(\S+)'
    model_path = re.search(pattern_path, test_log)
    return model_path.group(1) if model_path else 'Model was not found! Please contact with QA team'


def set_path_for_pytorch_files(instance, final_path):
    instance.ie_pipeline['prepare_model']['prepare_model_for_mo']['torch_model_zoo_path'] = final_path
    # if pytorch weights is required for tests we should use new path also for them
    weights_path = instance.ie_pipeline['prepare_model']['prepare_model_for_mo'].get('weights')
    if weights_path:
        weights_path = Path(weights_path)
        copied_weights_path = os.path.join(final_path, weights_path.parents[1].name,
                                           weights_path.parents[0].name, weights_path.name)
        instance.ie_pipeline['prepare_model']['prepare_model_for_mo']['weights'] = copied_weights_path
    return instance


@pytest.fixture(scope="function")
def copy_input_files(instance):
    """
    Fixture for coping model from shared folder to localhost.
    """
    pass
    # def wait_copy_finished(path_to_local_inputs, timeout=60):
    #     isCopied = False
    #     while timeout > 0:
    #         if os.path.exists(os.path.join(path_to_local_inputs, 'copy_complete')):
    #             isCopied = True
    #             break
    #         else:
    #             time.sleep(1)
    #             timeout -= 1
    #     return isCopied
    # if 'get_ovc_model' not in instance.ie_pipeline.get('get_ir', "None"):
    #     return
    # # define value to copy
    # prefix = os.path.join(instance.environment['input_model_dir'], '')
    # if not os.path.exists(prefix):
    #     os.mkdir(prefix)
    # if instance.ie_pipeline.get('load_pytorch_model') or instance.ie_pipeline.get('pytorch_to_onnx'):
    #     if instance.ie_pipeline.get('load_pytorch_model'):
    #         if instance.ie_pipeline['load_pytorch_model'].get('custom_pytorch_model_loader'):
    #             # it's hard to find out what to copy because it could be anything in that case
    #             model = None
    #         else:
    #             model = instance.ie_pipeline['load_pytorch_model']['load_pytorch_model'].get('model-path')
    #     if instance.ie_pipeline.get('pytorch_to_onnx'):
    #         model = instance.ie_pipeline['pytorch_to_onnx']['convert_pytorch_to_onnx'].get('model-path')
    #     # in that case we load model during the test so there is nothing to copy
    #     if not model:
    #         return
    # else:
    #     if isinstance(instance.ie_pipeline['get_ir']['get_ovc_model']['model'], str):
    #         model = Path(instance.ie_pipeline['get_ir']['get_ovc_model']['model'])
    #     else:
    #         return
    # model = Path(model)
    # if os.path.isfile(model):
    #     input_path = os.path.join(prefix, model.parents[1].name, model.parents[0].name)
    #     model_path = model.parent
    #     result_path = os.path.join(input_path, model.name)
    # else:
    #     input_path = os.path.join(prefix, model.parent.name, model.name)
    #     model_path = model
    #     result_path = input_path
    #
    # # copy stage
    # tries = 2
    # with log_timestamp('copy model'):
    #     for i in range(tries):
    #         try:
    #             shutil.copytree(model_path, input_path)
    #             open(os.path.join(input_path, 'copy_complete'), 'a').close()
    #             if instance.ie_pipeline.get('prepare_model'):
    #                 instance = set_path_for_pytorch_files(instance, result_path)
    #             else:
    #                 instance.ie_pipeline['get_ir']['get_ovc_model']['model'] = result_path
    #         except FileExistsError:
    #             if wait_copy_finished(input_path):
    #                 if instance.ie_pipeline.get('prepare_model'):
    #                     instance = set_path_for_pytorch_files(instance, result_path)
    #                 else:
    #                     instance.ie_pipeline['get_ir']['get_ovc_model']['model'] = result_path
    #         except BaseException:
    #             if i < tries - 1:
    #                 continue
    #             else:
    #                 raise
    #         break
    #     log.info(f'Input model was copied from {model} to {input_path}')
