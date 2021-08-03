# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""
Basic high-level plugin file for pytest.

See [Writing plugins](https://docs.pytest.org/en/latest/writing_plugins.html)
for more information.

This plugin adds the following command-line options:

* `--test_conf` - Path to test configuration file. Used to parametrize tests.
  Format: YAML file.
* `--exe` - Path to a test binary to execute.
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
import pytest
import yaml
from copy import deepcopy

from pathlib import Path
from jsonschema import validate, ValidationError

UTILS_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(UTILS_DIR))

from scripts.run_test import check_positive_int
from test_runner.utils import upload_data, metadata_from_manifest, query_timeline, DATABASES, DB_COLLECTIONS
from platform_utils import get_os_name, get_os_version, get_cpu_info


# -------------------- CLI options --------------------


def pytest_addoption(parser):
    """Specify command-line options for all plugins"""
    test_args_parser = parser.getgroup("test run")
    test_args_parser.addoption(
        "--test_conf",
        type=Path,
        help="path to a test config",
        default=Path(__file__).parent / "test_config.yml"
    )
    test_args_parser.addoption(
        "--exe",
        required=True,
        dest="executable",
        type=Path,
        help="path to a test binary to execute"
    )
    test_args_parser.addoption(
        "--niter",
        type=check_positive_int,
        help="number of iterations to run executable and aggregate results",
        default=3
    )
    helpers_args_parser = parser.getgroup("test helpers")
    helpers_args_parser.addoption(
        "--dump_refs",
        type=Path,
        help="path to dump test config with references updated with statistics collected while run",
    )
    helpers_args_parser.addoption(
        '--timeline_report',
        type=Path,
        # TODO:
        help='path to build manifest to extract commit information'
    )
    db_args_parser = parser.getgroup("test database use")
    db_args_parser.addoption(
        '--db_submit',
        metavar="RUN_ID",
        type=str,
        help='submit results to the database. ' \
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
        '--db_name',
        type=str,
        required=is_db_used,
        help='database name',
        choices=DATABASES
    )
    db_args_parser.addoption(
        '--db_collection',
        type=str,
        required=is_db_used,
        help='collection name in database',
        choices=DB_COLLECTIONS
    )
    db_args_parser.addoption(
        '--db_metadata',
        type=str,
        default=None,
        help='path to JSON-formatted file to extract additional information'
    )
    db_args_parser.addoption(
        '--manifest',
        type=Path,
        required=is_db_used,
        help='path to build manifest to extract commit information'
    )


@pytest.fixture(scope="session")
def executable(request):
    """Fixture function for command-line option."""
    return request.config.getoption('executable')


@pytest.fixture(scope="session")
def niter(request):
    """Fixture function for command-line option."""
    return request.config.getoption('niter')


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
    if instance["instance"]["device"]["name"] == "GPU":
        cl_cache_dir = pytestconfig.invocation_dir / "cl_cache"
        # if cl_cache generation to a local `cl_cache` folder doesn't work, specify
        # `cl_cache_dir` environment variable in an attempt to fix it (Linux specific)
        os.environ["cl_cache_dir"] = str(cl_cache_dir)
        if cl_cache_dir.exists():
            shutil.rmtree(cl_cache_dir)
        cl_cache_dir.mkdir()
        logging.info("cl_cache will be created in {}".format(cl_cache_dir))
        yield cl_cache_dir
        shutil.rmtree(cl_cache_dir)
    else:
        yield None


@pytest.fixture(scope="function")
def model_cache_dir(pytestconfig, instance):
    """
    Generate directory to IE model cache before test run and clean up after run.
    """
    if instance["instance"].get("use_model_cache"):
        model_cache_dir = pytestconfig.invocation_dir / "models_cache"
        if model_cache_dir.exists():
            shutil.rmtree(model_cache_dir)
        model_cache_dir.mkdir()
        logging.info("model_cache will be created in {}".format(model_cache_dir))
        yield model_cache_dir
        shutil.rmtree(model_cache_dir)
    else:
        yield None


@pytest.fixture(scope="function")
def validate_test_case(request):
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
        validate(instance=request.node.funcargs["instance"]["instance"], schema=schema)
    except ValidationError:
        request.config.option.db_submit = False
        raise
    yield


@pytest.fixture(scope="function")
def prepare_db_info(request, instance, executable, niter, manifest_metadata):
    """Fixture for preparing and validating data to submit to a database.

    Fixture prepares data and metadata to submit to a database. One of the steps
    is parsing of build information from build manifest. After preparation,
    it checks if data contains required properties.
    """
    FIELDS_FOR_ID = ['run_id', "test_exe", 'model', 'device', 'niter']

    run_id = request.config.getoption("db_submit")
    if not run_id:
        yield
        return

    # add db_metadata
    db_meta_path = request.config.getoption("db_metadata")
    if db_meta_path:
        with open(db_meta_path, "r") as db_meta_f:
            instance["db"].update(json.load(db_meta_f))

    # add test info
    info = {
        # results will be added immediately before uploading to DB in `pytest_runtest_makereport`.
        **instance["orig_instance"],  # TODO: think about use `instance` instead of `orig_instance`
        "run_id": run_id,
        "test_exe": str(executable.stem),
        "niter": niter,
        "test_name": request.node.name,
        "os": "_".join([str(item) for item in [get_os_name(), *get_os_version()]]),
        "cpu_info": get_cpu_info(),
        "status": "not_finished",
        "error_msg": "",
        "results": {},
        "raw_results": {},
        "references": instance["instance"].get("references", {}),   # upload actual references that were used
    }
    info['_id'] = hashlib.sha256(
        ''.join([str(info[key]) for key in FIELDS_FOR_ID]).encode()).hexdigest()
    instance["db"] = info

    # add manifest metadata
    instance["db"].update(manifest_metadata)

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
            "test_exe": {"type": "string"},
            "niter": {"type": "integer"},
            "test_name": {"type": "string"},
            "os": {"type": "string"},
            "cpu_info": {"type": "string"},
            "status": {"type": "string"},
            "error_msg": {"type": "string"},
            "results": {"type": "object"},
            "raw_results": {"type": "object"},
            "references": {"type": "object"},
            "_id": {"type": "string"}
        },
        "required": ["device", "model", "run_id", "test_exe", "niter", "test_name", "os", "cpu_info", 
                     "status", "error_msg", "results", "raw_results", "references", "_id"],
        "additionalProperties": true
    }
    """
    schema = json.loads(schema)

    try:
        validate(instance=instance["db"], schema=schema)
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


@pytest.fixture(scope="session", autouse=True)
def prepare_timeline_report(pytestconfig): #records, args.db_url, args.db_collection, args.timeline_report
    """ Create memcheck timeline HTML report for records.
    """
    yield
    report_path = pytestconfig.getoption('timeline_report')
    if report_path:
        db_url = pytestconfig.getoption("db_url")
        db_name = pytestconfig.getoption("db_name")
        db_collection = pytestconfig.getoption("db_collection")

        records = [rec["db"] for rec in pytestconfig.session_info]
        records.sort(
            key=lambda item: f"{item['status']}{item['device']['name']}{item['model']['name']}{item['test_name']}")
        timelines = query_timeline(records, db_url, db_name, db_collection)
        import jinja2  # pylint: disable=import-outside-toplevel
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(
                searchpath=Path().absolute() / 'memcheck-template'),
            autoescape=False)
        template = env.get_template('timeline_report.html')
        template.stream(records=records, timelines=timelines).dump(report_path)


@pytest.fixture(scope="session", autouse=True)
def prepare_tconf_with_refs(pytestconfig):
    """Fixture for preparing test config based on original test config
    with timetests results saved as references.
    """
    yield
    new_tconf_path = pytestconfig.getoption('dump_refs')
    if new_tconf_path:
        logging.info("Save new test config with test results as references to {}".format(new_tconf_path))
        upd_cases = pytestconfig.orig_cases.copy()
        for record in pytestconfig.session_info:
            rec_i = upd_cases.index(record["orig_instance"])
            upd_cases[rec_i]["references"] = record["results"]
        with open(new_tconf_path, "w") as tconf:
            yaml.safe_dump(upd_cases, tconf)


def pytest_generate_tests(metafunc):
    """Pytest hook for test generation.

    Generate parameterized tests from discovered modules and test config
    parameters.
    """
    with open(metafunc.config.getoption('test_conf'), "r") as file:
        test_cases = yaml.safe_load(file)
    if test_cases:
        test_cases = [{
            "instance": case,
            "orig_instance": deepcopy(case),
            "results": {}
        } for case in test_cases]
        metafunc.parametrize("instance", test_cases)
        setattr(metafunc.config, "session_info", test_cases)


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
    values = {key: val["instance"][key] for key in keys}
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

    db_url = item.config.getoption("db_url")
    db_name = item.config.getoption("db_name")
    db_collection = item.config.getoption("db_collection")

    instance = item.funcargs["instance"]  # alias
    report = (yield).get_result()
    if call.when in ["setup", "call"]:
        if call.when == "call":
            if not report.passed:
                instance["db"]["status"] = "failed"
                instance["db"]["error_msg"] = report.longrepr.reprcrash.message
            else:
                instance["db"]["status"] = "passed"
        instance["db"]["results"] = instance["results"]
        logging.info("Upload data to {}/{}.{}. Data: {}".format(db_url, db_name, db_collection, instance["db"]))
        # TODO: upload to new DB (memcheck -> memory_tests)
        #upload_data(data, db_url, db_name, db_collection)
