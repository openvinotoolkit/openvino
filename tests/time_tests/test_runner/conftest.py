# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""
Basic high-level plugin file for pytest.
See [Writing plugins](https://docs.pytest.org/en/latest/writing_plugins.html)
for more information.
This plugin adds the following command-line options:
* `--test_conf` - Path to test configuration file. Used to parametrize tests.
  Format: YAML file.
* `--exe` - Path to a timetest binary to execute.
* `--niter` - Number of times to run executable.
"""

import hashlib
import json
import logging
# pylint:disable=import-error
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest
import yaml
from jsonschema import validate, ValidationError

# add utils folder to imports
UTILS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "utils")
sys.path.insert(0, str(UTILS_DIR))

from path_utils import check_positive_int
from platform_utils import get_os_name, get_os_version, get_cpu_info
from utils import upload_data, metadata_from_manifest, push_to_db_facade, modify_data_for_push_to_new_db, DB_COLLECTIONS

# -------------------- CLI options --------------------


def pytest_addoption(parser):
    """Specify command-line options for all plugins"""
    test_args_parser = parser.getgroup("timetest test run")
    test_args_parser.addoption(
        "--test_conf",
        type=Path,
        help="Path to a test config",
        default=Path(__file__).parent / "test_config.yml"
    )
    test_args_parser.addoption(
        "--exe",
        required=True,
        dest="executable",
        type=Path,
        help="Path to a timetest binary to execute"
    )
    test_args_parser.addoption(
        "--niter",
        type=check_positive_int,
        help="Number of iterations to run executable and aggregate results",
        default=3
    )
    test_args_parser.addoption(
        "--model_cache",
        action='store_true',
        help="Enable model cache usage",
    )
    db_args_parser = parser.getgroup("timetest database use")
    db_args_parser.addoption(
        '--db_submit',
        metavar="RUN_ID",
        type=str,
        help='Submit results to the database. ' \
             '`RUN_ID` should be a string uniquely identifying the run' \
             ' (like Jenkins URL or time)'
    )
    is_db_used = db_args_parser.parser.parse_known_args(sys.argv).db_submit
    db_args_parser.addoption(
        '--db_url',
        type=str,
        required=is_db_used,
        help='MongoDB URL in a form "mongodb://server:port"'
    )
    db_args_parser.addoption(
        '--db_collection',
        type=str,
        required=is_db_used,
        help='Collection name in database',
        choices=DB_COLLECTIONS
    )
    db_args_parser.addoption(
        '--db_metadata',
        type=str,
        default=None,
        help='Path to JSON-formatted file to extract additional information'
    )
    db_args_parser.addoption(
        '--manifest',
        type=Path,
        required=is_db_used,
        help='Path to build manifest to extract commit information'
    )
    db_args_parser.addoption(
        '--db_api_handler',
        type=str,
        help='API handler url for push data to database',
        default=''
    )


@pytest.fixture(scope="session")
def test_conf(request):
    """Fixture function for command-line option."""
    return request.config.getoption('test_conf')


@pytest.fixture(scope="session")
def executable(request):
    """Fixture function for command-line option."""
    return request.config.getoption('executable')


@pytest.fixture(scope="session")
def niter(request):
    """Fixture function for command-line option."""
    return request.config.getoption('niter')


@pytest.fixture(scope="session")
def model_cache(request):
    """Fixture function for command-line option."""
    return request.config.getoption('model_cache')


# -------------------- CLI options --------------------


@pytest.fixture(scope="function")
def temp_dir(pytestconfig):
    """Create temporary directory for test purposes.
    It will be cleaned up after every test run.
    """
    temp_dir = tempfile.TemporaryDirectory()
    yield Path(temp_dir.name)
    temp_dir.cleanup()


@pytest.fixture(scope="function")
def cl_cache_dir(pytestconfig, instance):
    """Generate directory to save OpenCL cache before test run and clean up after run.
    Folder `cl_cache` should be created in a directory where tests were run. In this case
    cache will be saved correctly. This behaviour is OS independent.
    More: https://github.com/intel/compute-runtime/blob/master/opencl/doc/FAQ.md#how-can-cl_cache-be-enabled
    """
    if instance["device"]["name"] == "GPU":
        cl_cache_dir = pytestconfig.invocation_dir / "cl_cache"
        # if cl_cache generation to a local `cl_cache` folder doesn't work, specify
        # `cl_cache_dir` environment variable in an attempt to fix it (Linux specific)
        os.environ["cl_cache_dir"] = str(cl_cache_dir)
        if cl_cache_dir.exists():
            shutil.rmtree(cl_cache_dir)
        cl_cache_dir.mkdir()
        logging.info(f"cl_cache will be created in {cl_cache_dir}")
        yield cl_cache_dir
        shutil.rmtree(cl_cache_dir)
    else:
        yield None


@pytest.fixture(scope="function")
def model_cache_dir(pytestconfig, instance):
    """
    Generate directory to OV model cache before test run and clean up after run.
    """
    if instance.get("use_model_cache"):
        model_cache_dir = pytestconfig.invocation_dir / "models_cache"
        if model_cache_dir.exists():
            shutil.rmtree(model_cache_dir)
        model_cache_dir.mkdir()
        logging.info(f"model_cache will be created in {model_cache_dir}")
        yield model_cache_dir
        shutil.rmtree(model_cache_dir)
    else:
        yield None


@pytest.fixture(scope="function")
def test_info(request, pytestconfig):
    """Fixture for collecting timetests information.
    Current fixture fills in `request` and `pytestconfig` global
    fixtures with timetests information which will be used for
    internal purposes.
    """
    setattr(request.node._request, "test_info", {"results": {},
                                                 "raw_results": {},
                                                 "db_info": {}})

    yield request.node._request.test_info


@pytest.fixture(scope="function")
def validate_test_case(request, test_info):
    """Fixture for validating test case on correctness.
    Fixture checks current test case contains all fields required for
    a correct work.
    """
    schema = """
    {
        "type": "object",
        "properties": {
            "device": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"}
                },
                "required": ["name"]
            },
            "model": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"}
                },
                "required": ["path"]
            }
        },
        "required": ["device", "model"],
        "additionalProperties": true
    }
    """
    schema = json.loads(schema)

    try:
        validate(instance=request.node.funcargs["instance"], schema=schema)
    except ValidationError:
        request.config.option.db_submit = False
        raise
    yield


@pytest.fixture(scope="function")
def prepare_db_info(request, test_info, executable, niter, manifest_metadata):
    """Fixture for preparing and validating data to submit to a database.
    Fixture prepares data and metadata to submit to a database. One of the steps
    is parsing of build information from build manifest. After preparation,
    it checks if data contains required properties.
    """
    FIELDS_FOR_ID = ['run_id', 'timetest', 'model', 'device', 'niter']

    run_id = request.config.getoption("db_submit")
    if not run_id:
        yield
        return

    # add db_metadata
    db_meta_path = request.config.getoption("db_metadata")
    if db_meta_path:
        with open(db_meta_path, "r") as db_meta_f:
            test_info["db_info"].update(json.load(db_meta_f))

    # add model cache status
    test_info["db_info"].update({"model_cache": request.config.getoption("model_cache")})

    # add test info
    info = {
        # results will be added immediately before uploading to DB in `pytest_runtest_makereport`
        "run_id": run_id,
        "timetest": str(executable.stem),
        "model": request.node.funcargs["instance"]["model"],
        "device": request.node.funcargs["instance"]["device"],
        "niter": niter,
        "test_name": request.node.name,
        "os": "_".join([str(item) for item in [get_os_name(), *get_os_version()]])
    }
    info['_id'] = hashlib.sha256(
        ''.join([str(info[key]) for key in FIELDS_FOR_ID]).encode()).hexdigest()
    test_info["db_info"].update(info)

    # add manifest metadata
    test_info["db_info"].update(manifest_metadata)

    # validate db_info
    schema = """
    {
        "type": "object",
        "properties": {
            "device": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"}
                },
                "required": ["name"]
            },
            "model": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "name": {"type": "string"},
                    "precision": {"type": "string"},
                    "framework": {"type": "string"}
                },
                "required": ["path", "name", "precision", "framework"]
            },
            "run_id": {"type": "string"},
            "timetest": {"type": "string"},
            "niter": {"type": "integer"},
            "test_name": {"type": "string"},
            "results": {"type": "object"},
            "os": {"type": "string"},
            "_id": {"type": "string"}
        },
        "required": ["device", "model", "run_id", "timetest", "niter", "test_name", "os", "_id"],
        "additionalProperties": true
    }
    """
    schema = json.loads(schema)

    try:
        validate(instance=test_info["db_info"], schema=schema)
    except ValidationError:
        request.config.option.db_submit = False
        raise
    yield


@pytest.fixture(scope="session", autouse=True)
def manifest_metadata(request):
    """Fixture function for command-line option."""

    run_id = request.config.getoption("db_submit")
    if not run_id:
        yield
        return

    manifest_meta = metadata_from_manifest(request.config.getoption("manifest"))

    schema = """
        {
            "type": "object",
            "properties": {
                "product_type": {"type": "string"},
                "repo_url": {"type": "string"},
                "commit_sha": {"type": "string"},
                "commit_date": {"type": "string"},
                "branch": {"type": "string"},
                "target_branch": {"type": "string"},
                "version": {"type": "string"}
            },
            "required": ["product_type", "repo_url", "commit_sha", "commit_date", "branch", "target_branch", "version"],
            "additionalProperties": false
        }
        """
    schema = json.loads(schema)

    try:
        validate(instance=manifest_meta, schema=schema)
    except ValidationError:
        request.config.option.db_submit = False
        raise
    yield manifest_meta


def pytest_generate_tests(metafunc):
    """Pytest hook for test generation.
    Generate parameterized tests from discovered modules and test config
    parameters.
    """
    with open(metafunc.config.getoption('test_conf'), "r") as file:
        test_cases = yaml.safe_load(file)
    if test_cases:
        metafunc.parametrize("instance", test_cases)


def pytest_make_parametrize_id(config, val, argname):
    """Pytest hook for user-friendly test name representation"""

    def get_dict_values(d):
        """Unwrap dictionary to get all values of nested dictionaries"""
        if isinstance(d, dict):
            for v in d.values():
                yield from get_dict_values(v)
        else:
            yield d

    keys = ["device", "model"]
    values = {key: val[key] for key in keys}
    values = list(get_dict_values(values))

    return "-".join(["_".join([key, str(val)]) for key, val in zip(keys, values)])


@pytest.mark.hookwrapper
def pytest_runtest_makereport(item, call):
    """Pytest hook for report preparation.
    Submit tests' data to a database.
    """

    run_id = item.config.getoption("db_submit")
    if not run_id:
        yield
        return

    data = item._request.test_info["db_info"].copy()
    data["results"] = item._request.test_info["results"].copy()
    data["raw_results"] = item._request.test_info["raw_results"].copy()
    data["cpu_info"] = get_cpu_info()
    data["status"] = "not_finished"
    data["error_msg"] = ""

    report = (yield).get_result()
    if call.when in ["setup", "call"]:
        if call.when == "call":
            if not report.passed:
                data["status"] = "failed"
                data["error_msg"] = report.longrepr.reprcrash.message
            else:
                data["status"] = "passed"

        db_url = item.config.getoption("db_url")
        db_collection = item.config.getoption("db_collection")
        logging.info(f"Upload data to {db_url}/{'timetests'}.{db_collection}. "
                     f"Data: {data}")
        upload_data(data, db_url, 'timetests', db_collection)

        db_api_handler = item.config.getoption("db_api_handler")
        if db_api_handler and call.when == "call":
            new_format_records = modify_data_for_push_to_new_db(data)
            new_format_records['data'][0]["log"] = item._request.test_info["logs"]
            push_to_db_facade(new_format_records, db_api_handler)
