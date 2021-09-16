# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import numpy as np


def model_path(is_myriad=False):
    path_to_repo = os.environ["MODELS_PATH"]
    if not is_myriad:
        test_xml = os.path.join(path_to_repo, "models", "test_model", 'test_model_fp32.xml')
        test_bin = os.path.join(path_to_repo, "models", "test_model", 'test_model_fp32.bin')
    else:
        test_xml = os.path.join(path_to_repo, "models", "test_model", 'test_model_fp16.xml')
        test_bin = os.path.join(path_to_repo, "models", "test_model", 'test_model_fp16.bin')
    return (test_xml, test_bin)


def model_onnx_path():
    path_to_repo = os.environ["MODELS_PATH"]
    test_onnx = os.path.join(path_to_repo, "models", "test_model", 'test_model.onnx')
    return test_onnx

def image_path():
    path_to_repo = os.environ["DATA_PATH"]
    path_to_img = os.path.join(path_to_repo, 'validation_set', '224x224', 'dog.bmp')
    return path_to_img


def plugins_path():
    path_to_repo = os.environ["DATA_PATH"]
    plugins_xml = os.path.join(path_to_repo, 'ie_class', 'plugins.xml')
    plugins_win_xml = os.path.join(path_to_repo, 'ie_class', 'plugins_win.xml')
    plugins_osx_xml = os.path.join(path_to_repo, 'ie_class', 'plugins_apple.xml')
    return (plugins_xml, plugins_win_xml, plugins_osx_xml)


@pytest.fixture(scope='session')
def device():
    return os.environ.get("TEST_DEVICE") if os.environ.get("TEST_DEVICE") else "CPU"


def pytest_configure(config):
    # register an additional markers
    config.addinivalue_line(
        "markers", "ngraph_dependent_test"
    )
    config.addinivalue_line(
        "markers", "template_plugin"
    )


def create_ngraph_function(inputShape):
    import ngraph as ng
    inputShape = ng.impl.PartialShape(inputShape)
    param = ng.parameter(inputShape, dtype=np.float32, name="data")
    result = ng.relu(param, name='out')
    function  = ng.Function(result, [param], "TestFunction")
    return function
