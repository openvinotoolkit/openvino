# Copyright (C) 2020 Intel Corporation
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

# pylint:disable=import-error
import os
import sys
import pytest
from pathlib import Path
import yaml
import hashlib
import shutil
import logging

from test_runner.utils import upload_timetest_data, \
    DATABASE, DB_COLLECTIONS
from scripts.run_timetest import check_positive_int


# -------------------- CLI options --------------------


def pytest_addoption(parser):
    """Specify command-line options for all plugins"""
    test_args_parser = parser.getgroup("timetest test run")
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
        help="path to a timetest binary to execute"
    )
    test_args_parser.addoption(
        "--niter",
        type=check_positive_int,
        help="number of iterations to run executable and aggregate results",
        default=3
    )
    # TODO: add support of --mo, --omz etc. required for OMZ support
    helpers_args_parser = parser.getgroup("test helpers")
    helpers_args_parser.addoption(
        "--dump_refs",
        type=Path,
        help="path to dump test config with references updated with statistics collected while run",
    )
    db_args_parser = parser.getgroup("timetest database use")
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
        '--db_collection',
        type=str,
        required=is_db_used,
        help='collection name in "{}" database'.format(DATABASE),
        choices=DB_COLLECTIONS
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

# -------------------- CLI options --------------------


@pytest.fixture(scope="function")
def cl_cache_dir(pytestconfig):
    """Generate directory to save OpenCL cache before test run and clean up after run.

    Folder `cl_cache` should be created in a directory where tests were run. In this case
    cache will be saved correctly. This behaviour is OS independent.
    More: https://github.com/intel/compute-runtime/blob/master/opencl/doc/FAQ.md#how-can-cl_cache-be-enabled
    """
    cl_cache_dir = pytestconfig.invocation_dir / "cl_cache"
    # if cl_cache generation to a local `cl_cache` folder doesn't work, specify
    # `cl_cache_dir` environment variable in an attempt to fix it (Linux specific)
    os.environ["cl_cache_dir"] = str(cl_cache_dir)
    if cl_cache_dir.exists():
        shutil.rmtree(cl_cache_dir)
    cl_cache_dir.mkdir()
    yield cl_cache_dir
    shutil.rmtree(cl_cache_dir)


@pytest.fixture(scope="function")
def test_info(request, pytestconfig):
    """Fixture for collecting timetests information.

    Current fixture fills in `request` and `pytestconfig` global
    fixtures with timetests information which will be used for
    internal purposes.
    """
    setattr(request.node._request, "test_info", {"orig_instance": request.node.funcargs["instance"],
                                                 "results": {}})
    if not hasattr(pytestconfig, "session_info"):
        setattr(pytestconfig, "session_info", [])

    yield request.node._request.test_info

    pytestconfig.session_info.append(request.node._request.test_info)


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
        metafunc.parametrize("instance", test_cases)
        setattr(metafunc.config, "orig_cases", test_cases)


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

    FIELDS_FOR_ID = ['timetest', 'model', 'device', 'niter', 'run_id']
    FIELDS_FOR_SUBMIT = FIELDS_FOR_ID + ['_id', 'test_name',
                                         'results', 'status', 'error_msg']

    run_id = item.config.getoption("db_submit")
    db_url = item.config.getoption("db_url")
    db_collection = item.config.getoption("db_collection")
    if not (run_id and db_url and db_collection):
        yield
        return

    data = item.funcargs.copy()
    data["timetest"] = data.pop("executable").stem
    data.update(data["instance"])

    data['run_id'] = run_id
    data['_id'] = hashlib.sha256(
        ''.join([str(data[key]) for key in FIELDS_FOR_ID]).encode()).hexdigest()

    data["test_name"] = item.name
    data["results"] = item._request.test_info["results"]
    data["status"] = "not_finished"
    data["error_msg"] = ""

    data = {field: data[field] for field in FIELDS_FOR_SUBMIT}

    report = (yield).get_result()
    if call.when in ["setup", "call"]:
        if call.when == "call":
            if not report.passed:
                data["status"] = "failed"
                data["error_msg"] = report.longrepr.reprcrash.message
            else:
                data["status"] = "passed"
        logging.info("Upload data to {}/{}.{}. Data: {}".format(db_url, DATABASE, db_collection, data))
        upload_timetest_data(data, db_url, db_collection)
