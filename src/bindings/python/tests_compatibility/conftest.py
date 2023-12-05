# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import numpy as np

import ngraph as ng
import tests_compatibility

from pathlib import Path


def model_path(is_fp16=False):
    base_path = os.path.dirname(__file__)
    if is_fp16:
        test_xml = os.path.join(base_path, "test_utils", "utils", "test_model_fp16.xml")
        test_bin = os.path.join(base_path, "test_utils", "utils", "test_model_fp16.bin")
    else:
        test_xml = os.path.join(base_path, "test_utils", "utils", "test_model_fp32.xml")
        test_bin = os.path.join(base_path, "test_utils", "utils", "test_model_fp32.bin")
    return (test_xml, test_bin)


def model_onnx_path():
    base_path = os.path.dirname(__file__)
    test_onnx = os.path.join(base_path, "test_utils", "utils", "test_model.onnx")
    return test_onnx


def plugins_path():
    base_path = os.path.dirname(__file__)
    plugins_xml = os.path.join(base_path, "test_utils", "utils", "plugins.xml")
    plugins_win_xml = os.path.join(base_path, "test_utils", "utils", "plugins_win.xml")
    plugins_osx_xml = os.path.join(base_path, "test_utils", "utils", "plugins_apple.xml")
    return (plugins_xml, plugins_win_xml, plugins_osx_xml)


def _get_default_model_zoo_dir():
    return Path(os.getenv("ONNX_HOME", Path.home() / ".onnx/model_zoo"))


def pytest_addoption(parser):
    parser.addoption(
        "--backend",
        default="CPU",
        choices=["CPU", "GPU", "GNA", "HETERO", "TEMPLATE"],
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
    tests_compatibility.BACKEND_NAME = backend_name
    tests_compatibility.MODEL_ZOO_DIR = Path(config.getvalue("model_zoo_dir"))
    tests_compatibility.MODEL_ZOO_XFAIL = config.getvalue("model_zoo_xfail")

    # register additional markers
    config.addinivalue_line("markers", "skip_on_cpu: Skip test on CPU")
    config.addinivalue_line("markers", "skip_on_gpu: Skip test on GPU")
    config.addinivalue_line("markers", "skip_on_gna: Skip test on GNA")
    config.addinivalue_line("markers", "skip_on_hetero: Skip test on HETERO")
    config.addinivalue_line("markers", "skip_on_template: Skip test on TEMPLATE")
    config.addinivalue_line("markers", "onnx_coverage: Collect ONNX operator coverage")
    config.addinivalue_line("markers", "template_extension")
    config.addinivalue_line("markers", "dynamic_library: Runs tests only in dynamic libraries case")


def pytest_collection_modifyitems(config, items):
    backend_name = config.getvalue("backend")
    tests_compatibility.MODEL_ZOO_DIR = Path(config.getvalue("model_zoo_dir"))
    tests_compatibility.MODEL_ZOO_XFAIL = config.getvalue("model_zoo_xfail")

    keywords = {
        "CPU": "skip_on_cpu",
        "GPU": "skip_on_gpu",
        "GNA": "skip_on_gna",
        "HETERO": "skip_on_hetero",
        "TEMPLATE": "skip_on_template",
    }

    skip_markers = {
        "CPU": pytest.mark.skip(reason="Skipping test on the CPU backend."),
        "GPU": pytest.mark.skip(reason="Skipping test on the GPU backend."),
        "GNA": pytest.mark.skip(reason="Skipping test on the GNA backend."),
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


def create_encoder(input_shape, levels=4):
    # input
    input_node = ng.parameter(input_shape, np.float32, name="data")

    padding_begin = padding_end = [0, 0]
    strides = [1, 1]
    dilations = [1, 1]
    input_channels = [input_shape[1]]
    last_output = input_node

    # convolution layers
    for _ in range(levels):
        input_c = input_channels[-1]
        output_c = input_c * 2
        conv_w = np.random.uniform(0, 1, [output_c, input_c, 5, 5]).astype(np.float32)
        conv_node = ng.convolution(last_output, conv_w, strides, padding_begin, padding_end, dilations)
        input_channels.append(output_c)
        last_output = conv_node

    # deconvolution layers
    for _ in range(levels):
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
    function = ng.Function(result, [param], "TestFunction")
    return function
