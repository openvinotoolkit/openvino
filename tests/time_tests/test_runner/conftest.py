"""
Basic high-level plugin file for pytest.

See [Writing plugins](https://docs.pytest.org/en/latest/writing_plugins.html)
for more information.

This plugin adds the following command-line options:

* `--test_conf` - Path to test configuration file. Used to parametrize tests.
  Format: Python file.
* `--exe` - Path to a binary to execute.
* `--niter` - Number of iterations to run executable and aggregate results.
"""

# pylint:disable=import-error
import pytest
from pathlib import Path


# -------------------- CLI options --------------------

def pytest_addoption(parser):
    """Specify command-line options for all plugins"""
    parser.addoption(
        "--test_conf",
        type=Path,
        help="Path to test config",
        default=Path(".") / "test_config.py"
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
    # TODO: add support of --mo, --omz etc. required for OMZ support


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

    report = (yield).get_result()
    if call.when == "setup":
        # TODO: push all items to DB as "started"
        pass
    elif call.when == "call":
        # TODO: push all items to DB with some status
        exception_info = call.excinfo
