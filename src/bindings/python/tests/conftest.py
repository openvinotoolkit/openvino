# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest


def model_path(is_fp16=False):
    base_path = os.path.dirname(__file__)
    if is_fp16:
        test_xml = os.path.join(base_path, "utils", "utils", "test_model_fp16.xml")
        test_bin = os.path.join(base_path, "utils", "utils", "test_model_fp16.bin")
    else:
        test_xml = os.path.join(base_path, "utils", "utils", "test_model_fp32.xml")
        test_bin = os.path.join(base_path, "utils", "utils", "test_model_fp32.bin")
    return (test_xml, test_bin)


def model_onnx_path():
    base_path = os.path.dirname(__file__)
    test_onnx = os.path.join(base_path, "test_utils", "utils", "test_model.onnx")
    return test_onnx


def pytest_configure(config):

    # register additional markers
    config.addinivalue_line("markers", "skip_on_cpu: Skip test on CPU")
    config.addinivalue_line("markers", "skip_on_gpu: Skip test on GPU")
    config.addinivalue_line("markers", "skip_on_gna: Skip test on GNA")
    config.addinivalue_line("markers", "skip_on_hetero: Skip test on HETERO")
    config.addinivalue_line("markers", "skip_on_template: Skip test on TEMPLATE")
    config.addinivalue_line("markers", "onnx_coverage: Collect ONNX operator coverage")
    config.addinivalue_line("markers", "template_extension")
    config.addinivalue_line("markers", "dynamic_library: Runs tests only in dynamic libraries case")


@pytest.fixture(scope="session")
def device():
    return os.environ.get("TEST_DEVICE") if os.environ.get("TEST_DEVICE") else "CPU"
