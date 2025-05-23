# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest


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
