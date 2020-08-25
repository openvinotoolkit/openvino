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


def pytest_addoption(parser):
    """Specify command-line options for all plugins"""
    parser.addoption(
        "--test_conf",
        type=Path,
        help="Path to test config",
        default=Path(".") / "local_configs" / "test_config.xml"
    )
    parser.addoption(
        "--omz",
        type=Path,
        help="Path to Open Model Zoo root folder"
    )
    parser.addoption(
        "--mo",
        type=Path,
        help="Path to Model Optimizer main runner",
        default=Path(".") / ".." / ".." / "model_optimizer" / "mo.py"
    )
    parser.addoption(
        "--out_dir",
        type=Path,
        help="Path to the output directory to store models",
        default=Path(".") / "_out"
    )
    parser.addoption(
        '--no_venv',
        action="store_true",
        help='Skip preparation and use of virtual environment'
    )


@pytest.fixture(scope="session")
def test_conf(request):
    """Fixture function for command-line option."""
    return request.config.getoption('test_conf')


@pytest.fixture(scope="session")
def omz(request):
    """Fixture function for command-line option."""
    return request.config.getoption('omz')


@pytest.fixture(scope="session")
def mo(request):
    """Fixture function for command-line option."""
    return request.config.getoption('mo')


@pytest.fixture(scope="session")
def out_dir(request):
    """Fixture function for command-line option."""
    return request.config.getoption('out_dir')


@pytest.fixture(scope="session")
def no_venv(request):
    """Fixture function for command-line option."""
    return request.config.getoption('no_venv')











from xml.etree import ElementTree as ET
import itertools

def read_test_config(test_config):
    # TODO: unused
    """Read test configuration file and return cartesian product of found
    parameters (filtered and full).
    """
    test_conf_root = ET.parse(test_config).getroot()
    devices = []
    models = []
    for rec in test_conf_root.find("devices"):
        devices.append(rec.attrib)
    for rec in test_conf_root.find("models"):
        models.append(rec.attrib)

    items = []
    for item in itertools.product(devices, models):
        items.append({"device": item[0],
                      "model": item[1]})
    return items


def pytest_generate_tests(metafunc):
    """Pytest hook for test generation.

    Generate parameterized tests from discovered modules and test config
    parameters.
    """
    # TODO: attempt 1
    # test_cases = read_test_config(metafunc.config.getoption('test_conf'))
    # test_ids = []
    # # TODO: write!!!
    # # id = ""
    # # for case in test_cases:
    # #     id = "_".join([id] + ["{}-{}".format(key, val) for key, val in case.items()])
    #
    # if test_cases:
    #     # metafunc.parametrize("instance", test_cases, ids=test_ids)
    #     metafunc.parametrize("instance", test_cases)

    # TODO: attempt 2
    # test_conf_root = ET.parse(metafunc.config.getoption('test_conf')).getroot()
    # for group in ["device", "model"]:
    #     items = []
    #     for rec in test_conf_root.iter(group):
    #         items.append(rec.attrib)
    #     metafunc.parametrize(group, items)

    # TODO: attempt 3
    test_conf_root = ET.parse(metafunc.config.getoption('test_conf')).getroot()
    test_cases = []
    for test_rec in test_conf_root.iter("test"):
        test_case = {}
        for field in test_rec:
            test_case.update({field.tag: field.attrib})
        test_cases.append(test_case)
    if test_cases:
        # metafunc.parametrize("instance", test_cases, ids=test_ids)
        metafunc.parametrize("instance", test_cases)
