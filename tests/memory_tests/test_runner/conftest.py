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
* `--exe` - Path to a test binary to execute.
* `--niter` - Number of times to run executable.
"""

import hashlib
import json
import logging
# pylint:disable=import-error
import os
import pytest
import sys
import tempfile
import yaml
from copy import deepcopy
from inspect import getsourcefile
from jsonschema import validate, ValidationError
from pathlib import Path

UTILS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "utils")
sys.path.insert(0, str(UTILS_DIR))

from path_utils import check_positive_int
from proc_utils import cmd_exec
from platform_utils import get_os_name, get_os_version, get_cpu_info
from utils import metadata_from_manifest, DATABASES, DB_COLLECTIONS, upload_data

MEMORY_TESTS_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(MEMORY_TESTS_DIR)

from test_runner.utils import query_memory_timeline, REFS_FACTOR

OMZ_NUM_ATTEMPTS = 6


def abs_path(relative_path):
    """Return absolute path given path relative to the current file.
    """
    return os.path.realpath(
        os.path.join(os.path.dirname(getsourcefile(lambda: 0)), relative_path))


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
    omz_args_parser = parser.getgroup("Test with OMZ models")
    omz_args_parser.addoption(
        "--omz",
        type=Path,
        required=False,
        help="Path to Open Model Zoo (OMZ) repository root.",
    )
    omz_args_parser.addoption(
        "--omz_models_out_dir",
        type=Path,
        default=abs_path('../_omz_out/models'),
        help="Directory to put test data into.",
    )
    omz_args_parser.addoption(
        '--omz_cache_dir',
        type=Path,
        default=abs_path('../_omz_out/cache'),
        help='Directory with test data cache. Required for OMZ downloader.py only.'
    )
    omz_args_parser.addoption(
        '--omz_irs_out_dir',
        type=Path,
        default=abs_path('../_omz_out/irs'),
        help='Directory to put test data into. Required for OMZ converter.py only.'
    )
    helpers_args_parser = parser.getgroup("test helpers")
    helpers_args_parser.addoption(
        "--dump_refs",
        type=str,
        help="path to dump test config with references updated with statistics collected while run",
    )
    db_args_parser = parser.getgroup("test database use")
    db_args_parser.addoption(
        '--db_submit',
        metavar="RUN_ID",
        type=str,
        help='submit results to the database. '
             '`RUN_ID` should be a string uniquely identifying the run '
             '(like Jenkins URL or time)'
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
        '--manifest',
        type=Path,
        required=is_db_used,
        help='path to build manifest to extract commit information'
    )
    db_args_parser.addoption(
        '--db_metadata',
        type=str,
        default=None,
        help='path to JSON-formatted file to extract additional information'
    )
    db_args_parser.addoption(
        '--timeline_report',
        type=str,
        required=False,
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
def omz_models_conversion(instance, request):
    """
    Fixture for preparing omz models and updating test config with new paths
    """
    # Check Open Model Zoo key
    omz_path = request.config.getoption("omz")
    if omz_path:
        # TODO: After switch to wheel OV installation, omz tools should be accessible through command line
        downloader_path = omz_path / "tools" / "model_tools" / "downloader.py"
        converter_path = omz_path / "tools" / "model_tools" / "converter.py"
        info_dumper_path = omz_path / "tools" / "model_tools" / "info_dumper.py"

        if instance["instance"]["model"]["source"] == "omz":
            model_name = instance["instance"]["model"]["name"]
            model_precision = instance["instance"]["model"]["precision"]

            cache_dir = request.config.getoption("omz_cache_dir")
            omz_models_out_dir = request.config.getoption("omz_models_out_dir")
            omz_irs_out_dir = request.config.getoption("omz_irs_out_dir")

            # get full model info
            cmd = [f'{sys.executable}', f'{info_dumper_path}', '--name', f'{model_name}']
            return_code, info = cmd_exec(cmd, log=logging)
            assert return_code == 0, "Getting information about OMZ models has failed!"

            model_info = json.loads(info)[0]

            if model_precision not in model_info['precisions']:
                logging.error(f"Please specify precision for the model "
                              f"{model_name} from the list: {model_info['precisions']}")

            sub_model_path = str(Path(model_info["subdirectory"]) / model_precision / (model_name + ".xml"))
            model_out_path = omz_models_out_dir / sub_model_path
            model_irs_out_path = omz_irs_out_dir / sub_model_path

            # prepare models and convert models to IRs
            cmd = [f'{sys.executable}', f'{downloader_path}', '--name', f'{model_name}',
                   '--precisions', f'{model_precision}', '--num_attempts', f'{OMZ_NUM_ATTEMPTS}',
                   '--output_dir', f'{omz_models_out_dir}', '--cache_dir', f'{cache_dir}']

            return_code, _ = cmd_exec(cmd, log=logging)
            assert return_code == 0, "Downloading OMZ models has failed!"

            cmd = [f'{sys.executable}', f'{converter_path}', '--name', f'{model_name}', '-p', f'{sys.executable}',
                   '--precisions', f'{model_precision}', '--output_dir', f'{omz_irs_out_dir}',
                   '--download_dir', f'{omz_models_out_dir}']

            return_code, _ = cmd_exec(cmd, log=logging)
            assert return_code == 0, "Converting OMZ models has failed!"

            instance["instance"]["model"]["cache_path"] = model_out_path
            instance["instance"]["model"]["irs_out_path"] = model_irs_out_path


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
                    "name": {"type": "string"}
                },
                "required": ["name"]
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

    instance["db"] = {}

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
        "references": instance["instance"].get("references", {}),  # upload actual references that were used
        "ref_factor": REFS_FACTOR,
    }
    info['_id'] = hashlib.sha256(''.join([str(info[key]) for key in FIELDS_FOR_ID]).encode()).hexdigest()

    # add metadata
    instance["db"].update(info)
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
                    "name": {"type": "string"},
                    "precision": {"type": "string"},
                    "framework": {"type": "string"}
                },
                "required": ["name", "precision"]
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
    instance["db"]["results"] = instance["results"]
    instance["db"]["raw_results"] = instance["raw_results"]


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
def prepare_timeline_report(pytestconfig):
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

        timelines = query_memory_timeline(records, db_url, db_name, db_collection)

        import jinja2  # pylint: disable=import-outside-toplevel

        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(searchpath=os.path.join(abs_path('.'), 'memory_template')),
            autoescape=False)
        template = env.get_template('timeline_report.html')
        template.stream(records=records, timelines=timelines).dump(report_path)

        logging.info(f"Save html timeline_report to {report_path}")


@pytest.fixture(scope="session", autouse=True)
def prepare_tconf_with_refs(pytestconfig):
    """Fixture for preparing test config based on original test config
    with timetests results saved as references.
    """
    yield
    new_tconf_path = pytestconfig.getoption('dump_refs')
    if new_tconf_path:
        logging.info(f"Save new test config with test results as references to {new_tconf_path}")

        upd_cases = []
        steps_to_dump = {"create_exenetwork", "first_inference"}
        vm_metrics_to_dump = {"vmhwm", "vmrss"}
        stat_metrics_to_dump = {"avg"}

        for record in pytestconfig.session_info:
            rec_i = deepcopy(record["orig_instance"])
            rec_i["references"] = deepcopy(record["results"])

            for step_name, vm_records in rec_i["references"].copy().items():
                if step_name not in steps_to_dump:
                    del rec_i["references"][step_name]
                    continue
                for vm_metric, stat_metrics in vm_records.copy().items():
                    if vm_metric not in vm_metrics_to_dump:
                        del rec_i["references"][step_name][vm_metric]
                        continue
                    for stat_metric_name, _ in stat_metrics.copy().items():
                        if stat_metric_name not in stat_metrics_to_dump:
                            del rec_i["references"][step_name][vm_metric][stat_metric_name]
                            continue
            upd_cases.append(rec_i)

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
            "results": {},
            "raw_results": {},
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
    values = {key: val["instance"][key]["name"] for key in keys}
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
        instance["db"]["raw_results"] = instance["raw_results"]

        logging.info(f"Upload data to {db_url}/{db_name}.{db_collection}. "
                     f"Data: {instance['db']}")
        upload_data(instance["db"], db_url, db_name, db_collection)
