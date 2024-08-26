# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import pytest

import tests
import logging

from pathlib import Path


def _get_default_model_zoo_dir():
    return Path(os.getenv("ONNX_HOME", Path.home() / ".onnx/model_zoo"))


def pytest_addoption(parser):
    parser.addoption(
        "--backend",
        default="CPU",
        choices=["CPU", "GPU", "HETERO", "TEMPLATE"],
        help="Select target device",
    )
    parser.addoption(
        "--model_zoo_dir",
        default=_get_default_model_zoo_dir(),
        type=str,
        help="location of the model zoo",
    )
    parser.addoption(
        "--model_zoo_xfail",
        action="store_true",
        help="treat model zoo known issues as xfails instead of failures",
    )


def pytest_configure(config):
    backend_name = config.getvalue("backend")
    tests.BACKEND_NAME = backend_name
    tests.MODEL_ZOO_DIR = Path(config.getvalue("model_zoo_dir"))
    tests.MODEL_ZOO_XFAIL = config.getvalue("model_zoo_xfail")

    # register additional markers
    config.addinivalue_line("markers", "skip_on_cpu: Skip test on CPU")
    config.addinivalue_line("markers", "skip_on_gpu: Skip test on GPU")
    config.addinivalue_line("markers", "skip_on_hetero: Skip test on HETERO")
    config.addinivalue_line("markers", "skip_on_template: Skip test on TEMPLATE")
    config.addinivalue_line("markers", "onnx_coverage: Collect ONNX operator coverage")
    config.addinivalue_line("markers", "template_plugin")
    config.addinivalue_line("markers", "dynamic_library: Runs tests only in dynamic libraries case")

    # Issue 148922: trying to print what models
    # were found to debug test cases generation issue
    # Credits to https://stackoverflow.com/questions/36726461/how-to-print-output-when-using-pytest-with-xdist

    # Determine pytest-xdist worker id
    # Also see: https://pytest-xdist.readthedocs.io/en/latest/how-to.html#creating-one-log-file-for-each-worker
    worker_id = os.environ.get("PYTEST_XDIST_WORKER", default="gw0")

    # Create logs folder
    logs_folder = os.environ.get("LOGS_FOLDER", default="logs_folder")
    os.makedirs(logs_folder, exist_ok=True)

    # Create file handler to output logs into corresponding worker file
    file_handler = logging.FileHandler(f"{logs_folder}/logs_worker_{worker_id}.log", mode="w")
    file_handler.setFormatter(
        logging.Formatter(
            fmt="{asctime} {levelname}:{name}:{lineno}:{message}",
            style="{",
        )
    )

    # Create stream handler to output logs on console
    # This is a workaround for a known limitation:
    # https://pytest-xdist.readthedocs.io/en/latest/known-limitations.html
    console_handler = logging.StreamHandler(sys.stderr)  # pytest only prints error logs
    console_handler.setFormatter(
        logging.Formatter(
            # Include worker id in log messages, \r is needed to separate lines in console
            fmt="\r{asctime} " + worker_id + ":{levelname}:{name}:{lineno}:{message}",
            style="{",
        )
    )

    # Configure logging
    logging.basicConfig(level=logging.INFO, handlers=[console_handler, file_handler])

def pytest_collection_modifyitems(config, items):
    backend_name = config.getvalue("backend")
    tests.MODEL_ZOO_DIR = Path(config.getvalue("model_zoo_dir"))
    tests.MODEL_ZOO_XFAIL = config.getvalue("model_zoo_xfail")

    keywords = {
        "CPU": "skip_on_cpu",
        "GPU": "skip_on_gpu",
        "HETERO": "skip_on_hetero",
        "TEMPLATE": "skip_on_template",
    }

    skip_markers = {
        "CPU": pytest.mark.skip(reason="Skipping test on the CPU backend."),
        "GPU": pytest.mark.skip(reason="Skipping test on the GPU backend."),
        "HETERO": pytest.mark.skip(reason="Skipping test on the HETERO backend."),
        "TEMPLATE": pytest.mark.skip(reason="Skipping test on the TEMPLATE backend."),
    }

    for item in items:
        skip_this_backend = keywords[backend_name]
        if skip_this_backend in item.keywords:
            item.add_marker(skip_markers[backend_name])


@pytest.fixture(scope="session")
def device():
    return os.environ.get("TEST_DEVICE") if os.environ.get("TEST_DEVICE") else "CPU"
