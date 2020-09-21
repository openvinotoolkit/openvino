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
import pytest
from pathlib import Path
import yaml

from test_runner.utils import expand_env_vars


# -------------------- CLI options --------------------

def pytest_addoption(parser):
    """Specify command-line options for all plugins"""
    parser.addoption(
        "--test_conf",
        type=Path,
        help="Path to test config",
        default=Path(__file__).parent / "test_config.yml"
    )
    parser.addoption(
        "--exe",
        required=True,
        dest="executable",
        type=Path,
        help="Path to a timetest binary to execute",
    )
    parser.addoption(
        "--niter",
        type=int,
        help="Number of iterations to run executable and aggregate results",
        default=3
    )
    # TODO: add support of --mo, --omz etc. required for OMZ support


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


def pytest_generate_tests(metafunc):
    """Pytest hook for test generation.

    Generate parameterized tests from discovered modules and test config
    parameters.
    """
    with open(metafunc.config.getoption('test_conf'), "r") as file:
        test_cases = expand_env_vars(yaml.safe_load(file))
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

    keys = val.keys()
    values = list(get_dict_values(val))

    return "-".join(["_".join([key, val]) for key, val in zip(keys, values)])
