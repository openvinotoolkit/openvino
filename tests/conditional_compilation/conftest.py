#!/usr/bin/env python3
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=line-too-long

""" Pytest configuration for compilation tests.

Sample usage:
python3 -m pytest --artifacts ./compiled --models_root=<path to openvinotoolkit/testdata repository> \
    --sea_runtool=./IntelSEAPI/runtool/sea_runtool.py \
    --benchmark_app=./bin/benchmark_app test_collect.py
"""

import sys
from inspect import getsourcefile
from pathlib import Path

import pytest

# add ../lib to imports
sys.path.insert(
    0, str((Path(getsourcefile(lambda: 0)) / ".." / ".." / "lib").resolve(strict=True))
)

# Using models from https://github.com/openvinotoolkit/testdata
# $find models -wholename "*.xml"
TESTS = [
    {"path": "models/mobilenet_v2_1.4_224/mobilenet_v2_1.4_224_i8.xml"},
    {"path": "models/mobilenet_v2_1.0_224/mobilenet_v2_1.0_224_i8.xml"},
    {"path": "models/inception_v3/inception_v3_i8.xml"},
    {"path": "models/resnet_v1_50/resnet_v1_50_i8.xml"},
    {"path": "models/test_model/test_model_fp16.xml"},
    {"path": "models/test_model/test_model_fp32.xml"},
]


def pytest_addoption(parser):
    """ Define extra options for pytest options
    """
    parser.addoption(
        "--models_root", required=True, type=Path, help="Path to models root directory"
    )
    parser.addoption(
        "--sea_runtool", required=True, type=Path, help="Path to sea_runtool.py"
    )
    parser.addoption(
        "--benchmark_app",
        required=True,
        type=Path,
        help="Path to the benchmark_app tool",
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

    for test in TESTS:
        extra_args = {}
        path = test["path"]
        if "marks" in test:
            extra_args["marks"] = test["marks"]

        params.append(pytest.param(Path(path), **extra_args))
        ids = ids + [path]
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
def models_root(request):
    """Fixture function for command-line option."""
    return request.config.getoption("models_root")


@pytest.fixture(scope="session")
def artifacts(request):
    """Fixture function for command-line option."""
    return request.config.getoption("artifacts")
