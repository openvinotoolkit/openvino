#!/usr/bin/env python3

# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=line-too-long

""" Pytest configuration for compilation tests.

Sample usage:
python3 -m pytest --test_conf=<path to test config> \
    --sea_runtool=./thirdparty/itt_collector/runtool/sea_runtool.py --artifacts ./compiled test_collect.py \
    --collector_dir=./bin/intel64/Release --artifacts=<path to directory where tests write output or read input> \
    --openvino_ref=<Path to root directory with installed OpenVINO>
"""

import sys
import pytest
import yaml
from inspect import getsourcefile
from pathlib import Path

# add ../lib to imports
sys.path.insert(0, str((Path(getsourcefile(lambda: 0)) / ".." / ".." / "lib").resolve(strict=True)))

from path_utils import expand_env_vars  # pylint: disable=import-error


def pytest_addoption(parser):
    """Define extra options for pytest options."""
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
        "--collector_dir",
        type=Path,
        help="Path to a directory with a collector binary",
    )
    parser.addoption(
        "-A",
        "--artifacts",
        required=True,
        type=Path,
        help="Artifacts directory where tests write output or read input",
    )
    parser.addoption(
        "--openvino_ref",
        type=Path,
        help="Path to root directory with installed OpenVINO",
    )


def pytest_generate_tests(metafunc):
    """Generate tests depending on command line options."""
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
def collector_dir(request):
    """Fixture function for command-line option."""
    return request.config.getoption("collector_dir")


@pytest.fixture(scope="session")
def artifacts(request):
    """Fixture function for command-line option."""
    return request.config.getoption("artifacts")


@pytest.fixture(scope="session")
def openvino_root_dir(request):
    """Fixture function for command-line option."""
    return request.config.getoption("openvino_ref")
