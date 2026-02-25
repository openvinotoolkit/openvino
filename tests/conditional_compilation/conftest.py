#!/usr/bin/env python3

# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=line-too-long

"""Pytest configuration for compilation tests."""
import json
import logging
import sys
from inspect import getsourcefile
from pathlib import Path

# add utils folder to imports
sys.path.insert(0, str((Path(getsourcefile(lambda: 0)) / ".." / ".." / "utils").resolve(strict=True)))

import yaml
import pytest

from install_pkg import get_openvino_environment  # pylint: disable=import-error
from path_utils import expand_env_vars  # pylint: disable=import-error
from proc_utils import cmd_exec  # pylint: disable=import-error
from test_utils import make_build, validate_path_arg, write_session_info, \
    SESSION_INFO_FILE  # pylint: disable=import-error

log = logging.getLogger()
logging.basicConfig(format="[ %(name)s ] [ %(levelname)s ] %(message)s", level=logging.INFO, stream=sys.stdout)

OMZ_NUM_ATTEMPTS = 6


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
    parser.addoption(
        "--omz_repo",
        type=Path,
        default=Path("../_open_model_zoo").resolve(),
        help="Path to Open Model Zoo repository root directory",
    )
    parser.addoption(
        "--omz_cache_dir",
        type=Path,
        default=Path("../_omz_out/cache").resolve(),
        help="Path to Open Model Zoo cache directory",
    )


def pytest_generate_tests(metafunc):
    """Generate tests depending on command line options."""
    params = []
    ids = []

    with open(metafunc.config.getoption("test_conf"), "r") as file:
        test_cases = yaml.safe_load(file)

    for test in test_cases:
        model_list = []
        test_id_list = []
        for model in test:
            extra_args = {}
            if "marks" in test:
                extra_args["marks"] = test["marks"]
            model = model["model"]
            is_omz = model.get("type") == "omz"
            if is_omz:
                test_id_list.append(f'{model["name"]}_{model["precision"]}')
            else:
                test_id_list.append(model["path"].split("/")[-1])
                model["path"] = Path(expand_env_vars(model["path"]))
            model_list.append(model)
        ids = ids + ['-'.join(test_id_list)]
        params.append(pytest.param('-'.join(test_id_list), model_list), **extra_args)

    metafunc.parametrize("test_id, models", params, ids=ids)


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

    build_target = {"sea_itt_lib": Path(build_dir / "thirdparty" / "itt_collector" / "sea_itt_lib")}

    return_code, output = make_build(
        openvino_root_dir,
        build_dir,
        openvino_ref_path,
        build_target=build_target,
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


@pytest.fixture(scope="session")
def omz_repo(request):
    """Fixture function for command-line option."""
    omz_repo = request.config.getoption("omz_repo", skip=True)
    validate_path_arg(omz_repo, is_dir=True)

    return omz_repo


@pytest.fixture(scope="session")
def omz_cache_dir(request):
    """Fixture function for command-line option."""
    omz_cache_dir = request.config.getoption("omz_cache_dir", skip=True)
    if omz_cache_dir:
        try:
            validate_path_arg(omz_cache_dir, is_dir=True)
        except ValueError:
            log.warning(f'The Open Model Zoo cache directory'
                        f' "{omz_cache_dir}" does not exist.')

    return omz_cache_dir


@pytest.fixture(scope="function")
def prepared_models(openvino_ref, models, omz_repo, omz_cache_dir, tmpdir):
    """
    Process models: prepare Open Model Zoo models, skip non-OMZ models.
    """
    for model in models:
        if model.get("type") == "omz":
            model["path"] = prepare_omz_model(openvino_ref, model, omz_repo, omz_cache_dir, tmpdir)
    models = [model["path"] for model in models]
    return models


def prepare_omz_model(openvino_ref, model, omz_repo, omz_cache_dir, tmpdir):
    """
    Download and convert Open Model Zoo model to Intermediate Representation,
    get path to model XML.
    """
    omz_log = logging.getLogger("prepare_omz_model")

    python_executable = sys.executable
    converter_path = omz_repo / "tools" / "model_tools" / "converter.py"
    downloader_path = omz_repo / "tools" / "model_tools" / "downloader.py"
    info_dumper_path = omz_repo / "tools" / "model_tools" / "info_dumper.py"
    model_path_root = tmpdir

    # Step 1: downloader
    cmd = [f'{python_executable}', f'{downloader_path}',
           '--name', f'{model["name"]}',
           f'--precisions={model["precision"]}',
           '--num_attempts', f'{OMZ_NUM_ATTEMPTS}',
           '--output_dir', f'{model_path_root}']

    if omz_cache_dir:
        cmd.append('--cache_dir')
        cmd.append(f'{omz_cache_dir}')

    return_code, output = cmd_exec(cmd, log=omz_log)
    assert return_code == 0, "Downloading OMZ models has failed!"

    # Step 2: converter
    ir_path = model_path_root / "_IR"
    # Note: remove --precisions if both precisions (FP32 & FP16) are required
    cmd = [f'{python_executable}', f'{converter_path}',
           '--name', f'{model["name"]}',
           '-p', f'{python_executable}',
           f'--precisions={model["precision"]}',
           '--output_dir', f'{ir_path}',
           '--download_dir', f'{model_path_root}']

    return_code, output = cmd_exec(cmd, env=get_openvino_environment(openvino_ref), log=omz_log)
    assert return_code == 0, "Converting OMZ models has failed!"

    # Step 3: info_dumper
    cmd = [f'{python_executable}',
           f'{info_dumper_path}',
           '--name', f'{model["name"]}']

    return_code, output = cmd_exec(cmd, log=omz_log)
    assert return_code == 0, "Getting information about OMZ models has failed!"
    model_info = json.loads(output)[0]

    # Step 4: form model_path
    model_path = ir_path / model_info["subdirectory"] / model["precision"] / f'{model_info["name"]}.xml'

    return model_path
