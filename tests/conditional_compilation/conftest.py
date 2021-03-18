#!/usr/bin/env python3
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=line-too-long

""" Pytest configuration for compilation tests.

Sample usage:
python3 -m pytest --test_conf=<path to test config> \
    --sea_runtool=./IntelSEAPI/runtool/sea_runtool.py \
    --infer_tool=./tests/conditional_compilation/tools/infer_tool.py  --artifacts ./compiled test_collect.py \
    --collector_dir=./bin/intel64/Release --artifacts=<path to directory where tests write output or read input> \
    --install_dir=<path to full openvino installation dir> --install_cc_dir=<path to final openvino installation dir>
"""

import sys
from inspect import getsourcefile
from pathlib import Path

import pytest
import yaml

# add ../lib to imports
sys.path.insert(
    0, str((Path(getsourcefile(lambda: 0)) / ".." / ".." / "lib").resolve(strict=True))
)

from path_utils import expand_env_vars  # pylint: disable=import-error


def pytest_addoption(parser):
    """ Define extra options for pytest options
    """
    parser.addoption(
        "--test_conf",
        type=Path,
        default=Path(__file__).parent / "test_config.yml",
        help="Path to models root directory"
    )
    parser.addoption(
        "--sea_runtool",
        type=Path,
        help="Path to sea_runtool.py"
    )
    parser.addoption(
        "--infer_tool",
        type=Path,
        help="Path to the infer tool",
    )
    parser.addoption(
        "--collector_dir",
        type=Path,
        help="Path to a directory with a collector binary",
    )
    parser.addoption(
        "-A",
        "--artifacts",
        required=False,
        type=Path,
        help="Artifacts directory where tests write output or read input",
    )
    parser.addoption(
        "--install_dir",
        required=False,
        type=Path,
        help="Path to install directory before conditional compilation",
    )
    parser.addoption(
        "--install_cc_dir",
        required=False,
        type=Path,
        help="Path to install directory after conditional compilation",
    )


def pytest_generate_tests(metafunc):
    """ Generate tests depending on command line options
    """
    params = []
    ids = []

    with open(metafunc.config.getoption('test_conf'), "r") as file:
        test_cases = yaml.safe_load(file)

    for test in test_cases:
        extra_args = {}
        model_path = test["model"]["path"]
        if "marks" in test:
            extra_args["marks"] = test["marks"]

        test_id = model_path.replace('$', '').replace('{', '').replace('}', '')
        params.append(pytest.param(test_id, Path(expand_env_vars(model_path)), **extra_args))
        ids = ids + [test_id]
    metafunc.parametrize("test_id, model", params, ids=ids)


@pytest.fixture(scope="session")
def sea_runtool(request):
    """Fixture function for command-line option."""
    return request.config.getoption("sea_runtool")


@pytest.fixture(scope="session")
def infer_tool(request):
    """Fixture function for command-line option."""
    return request.config.getoption("infer_tool")


@pytest.fixture(scope="session")
def collector_dir(request):
    """Fixture function for command-line option."""
    return request.config.getoption("collector_dir")


@pytest.fixture(scope="session")
def artifacts(request):
    """Fixture function for command-line option."""
    return request.config.getoption("artifacts")


@pytest.fixture(scope="session")
def install_dir(request):
    """Fixture function for command-line option."""
    return request.config.getoption("install_dir")


@pytest.fixture(scope="session")
def install_cc_dir(request):
    """Fixture function for command-line option."""
    return request.config.getoption("install_cc_dir")
