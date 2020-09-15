"""
TODO: rewrite
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
"""

# pylint:disable=import-error
import pytest
from pathlib import Path
import yaml


def pytest_addoption(parser):
    """Specify command-line options for all plugins"""
    parser.addoption(
        "--test_conf",
        type=Path,
        help="Path to test config",
        default=Path(".") / "local_configs" / "test_config.yml"
    )
    parser.addoption(
        "--env_conf",
        type=Path,
        help="Path to environment config",
        default=Path(".") / "local_configs" / "env_config.yml"
    )
    parser.addoption(
        "--exe",
        required=True,
        dest="executable",
        type=Path,
        help="Path to a binary to execute",
    )
    parser.addoption(
        "--niter",
        type=int,
        help="Number of iterations to run executable and aggregate results",
        default=3
    )


@pytest.fixture(scope="session")
def test_conf(request):
    """Fixture function for command-line option."""
    return request.config.getoption('test_conf')


@pytest.fixture(scope="session")
def env_conf(request):
    """Fixture function for command-line option."""
    return request.config.getoption('env_conf')


@pytest.fixture(scope="session")
def executable(request):
    """Fixture function for command-line option."""
    return request.config.getoption('executable')


@pytest.fixture(scope="session")
def niter(request):
    """Fixture function for command-line option."""
    return request.config.getoption('niter')


class Environment:
    env = {}


def pytest_generate_tests(metafunc):
    """Pytest hook for test generation.

    Generate parameterized tests from discovered modules and test config
    parameters.
    """
    with open(metafunc.config.getoption('test_conf'), "r") as file:
        test_cases = yaml.load(file, Loader=yaml.FullLoader)
    if test_cases:
        metafunc.parametrize("instance", test_cases)


def pytest_make_parametrize_id(config, val, argname):
    return " {0}:{1} ".format(argname, val)


def pytest_sessionstart(session):
    with open(session.config.getoption('env_conf'), "r") as env_conf:
        Environment.env = yaml.load(env_conf, Loader=yaml.FullLoader)


@pytest.fixture(scope="function", autouse=True)
def db_report(request):
    # TODO: add reporting to DB before and after test
    instance = request.node.callspec.params["instance"]
    funcargs = request.node.funcargs
    exe = funcargs["executable"]
    niter = funcargs["niter"]
    env = Environment.env

    yield   # run test

    b = 2
