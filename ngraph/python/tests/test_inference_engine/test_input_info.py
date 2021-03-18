# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.inference_engine import IECore, IENetwork, ExecutableNetwork, InputInfoPtr, DataPtr, TensorDesc

import os
import pytest

def model_path(is_myriad=False):
    path_to_repo = os.environ["MODELS_PATH"]
    if not is_myriad:
        test_xml = os.path.join(path_to_repo, "models", "test_model", 'test_model_fp32.xml')
        test_bin = os.path.join(path_to_repo, "models", "test_model", 'test_model_fp32.bin')
    else:
        test_xml = os.path.join(path_to_repo, "models", "test_model", 'test_model_fp16.xml')
        test_bin = os.path.join(path_to_repo, "models", "test_model", 'test_model_fp16.bin')
    return (test_xml, test_bin)

test_net_xml, test_net_bin = model_path()


def get_input_info():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    return net.input_info["data"]


def test_input_info():
    assert isinstance(get_input_info(), InputInfoPtr)


def test_input_data():
    assert isinstance(get_input_info().input_data, DataPtr)


def test_input_data_setter():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    input_info = net.input_info["data"]
    other_input_data = net.outputs["fc_out"]
    input_info.input_data = other_input_data
    assert input_info.input_data.name == "fc_out"


def test_name():
    assert get_input_info().name == "data"


def test_precision():
    assert get_input_info().precision == "FP32"


def test_precision_setter():
    input_info = get_input_info()
    input_info.precision = "I8"
    assert input_info.precision == "I8", "Incorrect precision"


def test_layout():
    assert get_input_info().layout == "NCHW"


def test_layout_setter():
    input_info = get_input_info()
    input_info.layout = "NHWC"
    assert input_info.layout == "NHWC", "Incorrect layout"


def test_tensor_desc():
    tensor_desc = get_input_info().tensor_desc
    assert isinstance(tensor_desc, TensorDesc)
    assert tensor_desc.layout == "NCHW"
