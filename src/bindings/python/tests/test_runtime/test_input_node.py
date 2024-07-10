# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import numpy as np
from openvino.runtime import Input, RTMap
from openvino._pyopenvino import DescriptorTensor
import openvino.runtime.opset13 as ops

from openvino import Core, OVAny, Shape, PartialShape, Type, Tensor, Symbol
from tests.utils.helpers import get_relu_model


def test_input_type(device):
    core = Core()
    model = get_relu_model()
    compiled_model = core.compile_model(model, device)
    net_input = compiled_model.output(0)
    input_node = net_input.get_node().inputs()[0]
    assert isinstance(input_node, Input)


def test_const_output_docs(device):
    core = Core()
    model = get_relu_model()
    compiled_model = core.compile_model(model, device)
    net_input = compiled_model.output(0)
    input_node = net_input.get_node().inputs()[0]
    exptected_string = "openvino.runtime.Input wraps ov::Input<Node>"
    assert input_node.__doc__ == exptected_string


def test_input_get_index(device):
    core = Core()
    model = get_relu_model()
    compiled_model = core.compile_model(model, device)
    net_input = compiled_model.output(0)
    input_node = net_input.get_node().inputs()[0]
    assert input_node.get_index() == 0


def test_input_element_type(device):
    core = Core()
    model = get_relu_model()
    compiled_model = core.compile_model(model, device)
    net_input = compiled_model.output(0)
    input_node = net_input.get_node().inputs()[0]
    assert input_node.get_element_type() == Type.f32


def test_input_get_shape(device):
    core = Core()
    model = get_relu_model()
    compiled_model = core.compile_model(model, device)
    net_input = compiled_model.output(0)
    input_node = net_input.get_node().inputs()[0]
    assert str(input_node.get_shape()) == str(Shape([1, 3, 32, 32]))


def test_input_get_partial_shape(device):
    core = Core()
    model = get_relu_model()
    compiled_model = core.compile_model(model, device)
    net_input = compiled_model.output(0)
    input_node = net_input.get_node().inputs()[0]
    expected_partial_shape = PartialShape([1, 3, 32, 32])
    assert input_node.get_partial_shape() == expected_partial_shape


def test_input_get_source_output(device):
    core = Core()
    model = get_relu_model()
    compiled_model = core.compile_model(model, device)
    net_input = compiled_model.output(0)
    input_node = net_input.get_node().inputs()[0]
    name = input_node.get_source_output().get_node().get_friendly_name()
    assert name == "relu"


def test_input_get_tensor(device):
    core = Core()
    model = get_relu_model()
    compiled_model = core.compile_model(model, device)
    net_input = compiled_model.output(0)
    input_node = net_input.get_node().inputs()[0]
    tensor = input_node.get_tensor()
    assert isinstance(tensor, DescriptorTensor)


def test_input_get_rt_info(device):
    core = Core()
    model = get_relu_model()
    compiled_model = core.compile_model(model, device)
    net_input = compiled_model.output(0)
    input_node = net_input.get_node().inputs()[0]
    rt_info = input_node.get_rt_info()
    assert isinstance(rt_info, RTMap)


def test_input_rt_info(device):
    core = Core()
    model = get_relu_model()
    compiled_model = core.compile_model(model, device)
    net_input = compiled_model.output(0)
    input_node = net_input.get_node().inputs()[0]
    rt_info = input_node.rt_info
    assert isinstance(rt_info, RTMap)


def test_input_replace_source_output(device):
    param = ops.parameter([1, 64], Type.i64)
    param.output(0).get_tensor().set_names({"a", "b"})

    param1 = ops.parameter([1, 64], Type.i64)
    param1.output(0).get_tensor().set_names({"c", "d"})

    relu = ops.relu(param)
    relu.input(0).replace_source_output(param1.output(0))

    assert param.output(0).get_tensor().get_names() == {"a", "b"}
    assert param1.output(0).get_tensor().get_names() == {"c", "d"}


def test_input_update_rt_info(device):
    core = Core()
    model = get_relu_model()
    compiled_model = core.compile_model(model, device)
    net_input = compiled_model.output(0)
    input_node = net_input.get_node().inputs()[0]
    rt = input_node.get_rt_info()
    rt["test12345"] = "test"
    for key, value in input_node.get_rt_info().items():
        assert key == "test12345"
        assert isinstance(value, OVAny)


def test_tensor_bounds_in_model(device):
    core = Core()
    model = get_relu_model()
    compiled_model = core.compile_model(model, device)
    tensor = compiled_model.output(0).get_tensor()
    partial_shape = tensor.get_partial_shape().to_shape()
    tensor_size = np.prod(partial_shape)

    lower_value = np.zeros(tensor_size, dtype=np.float32)
    lower_value_tensor = Tensor(lower_value.reshape(partial_shape))
    upper_value = np.ones(tensor_size, dtype=np.float32)
    upper_value_tensor = Tensor(upper_value.reshape(partial_shape))

    tensor.set_lower_value(lower_value_tensor)
    retrieved_lower_value = tensor.get_lower_value().data
    tensor.set_upper_value(upper_value_tensor)
    retrieved_upper_value = tensor.get_upper_value().data
    assert np.array_equal(retrieved_lower_value, lower_value_tensor.data)
    assert np.array_equal(retrieved_upper_value, upper_value_tensor.data)


def test_value_symbol_in_model(device):
    core = Core()
    model = get_relu_model()
    compiled_model = core.compile_model(model, device)
    tensor = compiled_model.output(0).get_tensor()
    partial_shape = tensor.get_partial_shape().to_shape()
    tensor_size = np.prod(partial_shape)
    values = [Symbol() for _ in range(tensor_size)]
    tensor.set_value_symbol(values)
    assert tensor.get_value_symbol() == values
