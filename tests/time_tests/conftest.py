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


# -------------------- CLI options --------------------

def pytest_addoption(parser):
    """Specify command-line options for all plugins"""
    parser.addoption(
        "--test_conf",
        type=Path,
        help="Path to test config",
        default=Path(".") / "local_configs" / "test_config.py"
        # TODO: test config in Python vs. .yml to support xfails ???
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
    # TODO: Parse args from a file via argparse (to avoid env_config)


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

# -------------------- CLI options --------------------


class Environment:
    """Class responsible for managing environment information."""
    env = {}


def pytest_generate_tests(metafunc):
    """Pytest hook for test generation.

    Generate parameterized tests from discovered modules and test config
    parameters.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("test_config", metafunc.config.getoption('test_conf').resolve())
    test_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_config)
    test_cases = test_config.test_cases

    if test_cases:
        metafunc.parametrize("instance", test_cases)


def pytest_make_parametrize_id(config, val, argname):
    """Pytest hook for user-friendly test name representation"""
    return " {0}:{1} ".format(argname, val)


def pytest_sessionstart(session):
    """Pytest hook for session preparation.

    Fill `Environment` global with information from environment config.
    """
    with open(session.config.getoption('env_conf'), "r") as env_conf:
        Environment.env = yaml.load(env_conf, Loader=yaml.FullLoader)


@pytest.mark.hookwrapper
def pytest_runtest_makereport(item, call):
    """Pytest hook for report preparation.

    Report tests' data to DB.
    """
    test_name = item.name
    funcargs = item.funcargs
    instance = funcargs["instance"]
    exe = funcargs["executable"]
    niter = funcargs["niter"]
    env = Environment.env

    report = (yield).get_result()
    if call.when == "setup":
        # TODO: push all items to DB as "started"
        pass
    elif call.when == "call":
        # TODO: push all items to DB with some status
        exception_info = call.excinfo