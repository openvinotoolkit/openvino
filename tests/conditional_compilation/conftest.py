#!/usr/bin/env python3

# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=line-too-long

"""Pytest configuration for compilation tests."""

import logging
import sys
from inspect import getsourcefile
from pathlib import Path

# add ../lib to imports
sys.path.insert(0, str((Path(getsourcefile(lambda: 0)) / ".." / ".." / "lib").resolve(strict=True)))

import yaml
import pytest

from path_utils import expand_env_vars  # pylint: disable=import-error
from test_utils import make_build, validate_path_arg, write_session_info, SESSION_INFO_FILE  # pylint: disable=import-error


log = logging.getLogger()


def pytest_addoption(parser):
    """Define extra options for pytest options."""
    parser.addoption(
        "--test_conf",
        type=Path,
        default=Path(__file__).parent / "test_config.yml",
        help="Path to models root directory",
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
    parser.addoption(
        "--openvino_root_dir",
        type=Path,
        help="Path to OpenVINO repository root directory",
    )


def pytest_generate_tests(metafunc):
    """Generate tests depending on command line options."""
    params = []
    ids = []

    with open(metafunc.config.getoption("test_conf"), "r") as file:
        test_cases = yaml.safe_load(file)

    for test in test_cases:
        extra_args = {}
        model_path = test["model"]["path"]
        if "marks" in test:
            extra_args["marks"] = test["marks"]

        test_id = model_path.replace("$", "").replace("{", "").replace("}", "")
        params.append(pytest.param(test_id, Path(expand_env_vars(model_path)), **extra_args))
        ids = ids + [test_id]
    metafunc.parametrize("test_id, model", params, ids=ids)


@pytest.fixture(scope="session")
def sea_runtool(request):
    """Fixture function for command-line option."""
    sea_runtool = request.config.getoption("sea_runtool", skip=True)
    validate_path_arg(sea_runtool)

    return sea_runtool


@pytest.fixture(scope="session")
def collector_dir(request):
    """Fixture function for command-line option."""
    collector_dir = request.config.getoption("collector_dir", skip=True)
    validate_path_arg(collector_dir, is_dir=True)

    return collector_dir


@pytest.fixture(scope="session")
def artifacts(request):
    """Fixture function for command-line option."""
    return request.config.getoption("artifacts")


@pytest.fixture(scope="session")
def openvino_root_dir(request):
    """Fixture function for command-line option."""
    openvino_root_dir = request.config.getoption("openvino_root_dir", skip=True)
    validate_path_arg(openvino_root_dir, is_dir=True)

    return openvino_root_dir


@pytest.fixture(scope="session")
def openvino_ref(request, artifacts):
    """Fixture function for command-line option.
    Return path to root directory with installed OpenVINO.
    If --openvino_ref command-line option is not specified firstly build and install
    instrumented package with OpenVINO repository specified in --openvino_root_dir option.
    """
    openvino_ref = request.config.getoption("openvino_ref")
    if openvino_ref:
        validate_path_arg(openvino_ref, is_dir=True)

        return openvino_ref

    openvino_root_dir = request.config.getoption("openvino_root_dir", skip=True)
    validate_path_arg(openvino_root_dir, is_dir=True)

    build_dir = openvino_root_dir / "build_instrumented"
    openvino_ref_path = artifacts / "ref_pkg"

    log.info("--openvino_ref is not specified. Preparing instrumented build at %s", build_dir)

    return_code, output = make_build(
        openvino_root_dir,
        build_dir,
        openvino_ref_path,
        cmake_additional_args=["-DSELECTIVE_BUILD=COLLECT"],
        log=log
    )
    assert return_code == 0, f"Command exited with non-zero status {return_code}:\n {output}"

    return openvino_ref_path


@pytest.fixture(scope="function")
def test_info(request, pytestconfig):
    """Fixture function for getting the additional attributes of the current test."""
    setattr(request.node._request, "test_info", {})
    if not hasattr(pytestconfig, "session_info"):
        setattr(pytestconfig, "session_info", [])

    yield request.node._request.test_info

    pytestconfig.session_info.append(request.node._request.test_info)


@pytest.fixture(scope="session")
def save_session_info(pytestconfig, artifacts):
    """Fixture function for saving additional attributes to configuration file."""
    yield
    write_session_info(path=artifacts / SESSION_INFO_FILE, data=pytestconfig.session_info)
