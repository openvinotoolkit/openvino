#!/usr/bin/env python3
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=line-too-long

""" Pytest configuration for compilation tests.

Sample usage:
python3 -m pytest --artifacts ./compiled --test_conf=<path to test config> \
    --sea_runtool=./IntelSEAPI/runtool/sea_runtool.py \
    --benchmark_app=./bin/benchmark_app test_collect.py
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
        required=True,
        type=Path,
        help="Path to sea_runtool.py"
    )
    parser.addoption(
        "--benchmark_app",
        required=True,
        type=Path,
        help="Path to the benchmark_app tool",
    )
    parser.addoption(
        "--collector_dir",
        required=True,
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

        params.append(pytest.param(Path(expand_env_vars(model_path)), **extra_args))
        ids = ids + [model_path]
    metafunc.parametrize("model", params, ids=ids)


@pytest.fixture(scope="session")
def sea_runtool(request):
    """Fixture function for command-line option."""
    return request.config.getoption("sea_runtool")


@pytest.fixture(scope="session")
def benchmark_app(request):
    """Fixture function for command-line option."""
    return request.config.getoption("benchmark_app")


@pytest.fixture(scope="session")
def collector_dir(request):
    """Fixture function for command-line option."""
    return request.config.getoption("collector_dir")


@pytest.fixture(scope="session")
def artifacts(request):
    """Fixture function for command-line option."""
    return request.config.getoption("artifacts")
