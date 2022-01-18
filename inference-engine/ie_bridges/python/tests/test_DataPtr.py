# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from openvino.inference_engine import IECore, DataPtr
from conftest import model_path, create_relu
import ngraph as ng


test_net_xml, test_net_bin = model_path()


def layer_out_data():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    return net.outputs['fc_out']


def test_name():
    assert layer_out_data().name == 'fc_out', "Incorrect name for layer 'fc_out'"


def test_precision():
    assert layer_out_data().precision == "FP32", "Incorrect precision for layer 'fc_out'"


def test_precision_setter():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    net.outputs['fc_out'].precision = "I8"
    assert net.outputs['fc_out'].precision == "I8", "Incorrect precision for layer 'fc_out'"


def test_incorrect_precision_setter():
    with pytest.raises(ValueError) as e:
        layer_out_data().precision = "123"
    assert "Unsupported precision 123! List of supported precisions:" in str(e.value)


def test_layout():
    assert layer_out_data().layout == "NC", "Incorrect layout for layer 'fc_out'"


def test_initialized():
    assert layer_out_data().initialized, "Incorrect value for initialized property for layer 'fc_out'"


@pytest.mark.template_plugin
def test_is_dynamic():
    function = create_relu([-1, 3, 20, 20])
    net = ng.function_to_cnn(function)
    assert net.input_info["data"].input_data.is_dynamic
    assert net.outputs["out"].is_dynamic
    p_shape = ng.partial_shape_from_data(net.input_info["data"].input_data)
    assert isinstance(p_shape, ng.impl.PartialShape)
    p_shape = ng.partial_shape_from_data(net.outputs["out"])
    assert isinstance(p_shape, ng.impl.PartialShape)
    with pytest.raises(RuntimeError) as e:
        net.input_info["data"].input_data.shape
    assert  "Cannot return dims for Data with dynamic shapes!" in str(e.value)
    ie = IECore()
    ie.register_plugin("ov_template_plugin", "TEMPLATE")
    exec_net = ie.load_network(net, "TEMPLATE")
    assert exec_net.input_info["data"].input_data.is_dynamic
    p_shape = ng.partial_shape_from_data(exec_net.input_info["data"].input_data)
    assert isinstance(p_shape, ng.impl.PartialShape)
