# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import platform

import pytest

# Workaround for NumPy 2.x on RPi ARMv8.0 CPUs, ticket 179098
# https://numpy.org/devdocs/reference/simd/build-options.html
_npy_cpu_features_original = os.environ.get("NPY_DISABLE_CPU_FEATURES")
if platform.machine() == "aarch64" and platform.system() == "Linux":
    os.environ["NPY_DISABLE_CPU_FEATURES"] = "ASIMDDP,ASIMDFHM"


def pytest_sessionfinish(session, exitstatus):
    """Restore NPY_DISABLE_CPU_FEATURES after test session completes."""
    if _npy_cpu_features_original is None:
        os.environ.pop("NPY_DISABLE_CPU_FEATURES", None)
    else:
        os.environ["NPY_DISABLE_CPU_FEATURES"] = _npy_cpu_features_original


def pytest_configure(config):

    # register additional markers
    config.addinivalue_line("markers", "skip_on_cpu: Skip test on CPU")
    config.addinivalue_line("markers", "skip_on_gpu: Skip test on GPU")
    config.addinivalue_line("markers", "skip_on_hetero: Skip test on HETERO")
    config.addinivalue_line("markers", "skip_on_template: Skip test on TEMPLATE")
    config.addinivalue_line("markers", "onnx_coverage: Collect ONNX operator coverage")
    config.addinivalue_line("markers", "template_extension")
    config.addinivalue_line("markers", "dynamic_library: Runs tests only in dynamic libraries case")


@pytest.fixture(scope="session")
def device():
    return os.environ.get("TEST_DEVICE") if os.environ.get("TEST_DEVICE") else "CPU"
