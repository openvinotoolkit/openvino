# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import numpy as np

import ngraph as ng


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
    config.addinivalue_line("markers", "template_plugin: Skip test on Template plugin")
    config.addinivalue_line("markers", "dynamic_library: Runs tests only in dynamic libraries case")


def create_encoder(input_shape, levels = 4):
    # input
    input_node = ng.parameter(input_shape, np.float32, name="data")

    padding_begin = padding_end = [0, 0]
    strides = [1, 1]
    dilations = [1, 1]
    input_channels = [input_shape[1]]
    last_output = input_node

    # convolution layers
    for i in range(levels):
        input_c = input_channels[-1]
        output_c = input_c * 2
        conv_w = np.random.uniform(0, 1, [output_c, input_c, 5, 5]).astype(np.float32)
        conv_node = ng.convolution(last_output, conv_w, strides, padding_begin, padding_end, dilations)
        input_channels.append(output_c)
        last_output = conv_node

    # deconvolution layers
    for i in range(levels):
        input_c = input_channels[-2]
        output_c = input_channels.pop(-1)
        deconv_w = np.random.uniform(0, 1, [output_c, input_c, 5, 5]).astype(np.float32)
        deconv_node = ng.convolution_backprop_data(last_output, deconv_w, strides)
        last_output = deconv_node

    # result
    last_output.set_friendly_name("out")
    result_node = ng.result(last_output)
    return ng.Function(result_node, [input_node], "Encoder")


def create_relu(input_shape):
    input_shape = ng.impl.PartialShape(input_shape)
    param = ng.parameter(input_shape, dtype=np.float32, name="data")
    result = ng.relu(param, name="out")
    function  = ng.Function(result, [param], "TestFunction")
    return function
