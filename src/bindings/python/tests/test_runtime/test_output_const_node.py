# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from copy import copy, deepcopy

import openvino.opset13 as ops
from openvino import (
    Shape,
    PartialShape,
    Type,
    Core,
    OVAny,
)
from openvino import (
    ConstOutput,
    Output,
    RTMap,
)
from tests.utils.helpers import get_relu_model


def test_const_output_type(device):
    core = Core()
    model = get_relu_model()
    compiled_model = core.compile_model(model, device)
    node = compiled_model.input(0)
    assert isinstance(node, ConstOutput)


def test_const_output_docs(device):
    core = Core()
    model = get_relu_model()
    compiled_model = core.compile_model(model, device)
    node = compiled_model.input(0)
    exptected_string = "openvino.ConstOutput represents port/node output."
    assert node.__doc__ == exptected_string


def test_const_output_get_index(device):
    core = Core()
    model = get_relu_model()
    compiled_model = core.compile_model(model, device)
    node = compiled_model.input("data")
    assert node.get_index() == 0
    assert node.index == 0


def test_const_output_get_element_type(device):
    core = Core()
    model = get_relu_model()
    compiled_model = core.compile_model(model, device)
    node = compiled_model.input("data")
    assert node.get_element_type() == Type.f32
    assert node.element_type == Type.f32


def test_const_output_get_shape(device):
    core = Core()
    model = get_relu_model()
    compiled_model = core.compile_model(model, device)
    node = compiled_model.input("data")
    expected_shape = Shape([1, 3, 32, 32])
    assert str(node.get_shape()) == str(expected_shape)
    assert str(node.shape) == str(expected_shape)


def test_const_output_get_partial_shape(device):
    core = Core()
    model = get_relu_model()
    exec_net = core.compile_model(model, device)
    node = exec_net.input("data")
    expected_partial_shape = PartialShape([1, 3, 32, 32])
    assert node.get_partial_shape() == expected_partial_shape
    assert node.partial_shape == expected_partial_shape


def test_const_output_get_target_inputs(device):
    core = Core()
    model = get_relu_model()
    compiled_model = core.compile_model(model, device)
    outputs = compiled_model.outputs
    for node in outputs:
        assert isinstance(node.get_target_inputs(), set)
        assert isinstance(node.target_inputs, set)


def test_const_output_get_names(device):
    core = Core()
    model = get_relu_model()
    compiled_model = core.compile_model(model, device)
    input_name = "data"
    node = compiled_model.input(input_name)
    expected_names = set()
    expected_names.add(input_name)
    assert node.get_names() == expected_names
    assert node.names == expected_names
    assert node.get_any_name() == input_name
    assert node.any_name == input_name


def test_const_get_rf_info(device):
    core = Core()
    model = get_relu_model()
    compiled_model = core.compile_model(model, device)
    output_node = compiled_model.output(0)
    rt_info = output_node.get_rt_info()
    assert isinstance(rt_info, RTMap)


def test_const_output_runtime_info(device):
    core = Core()
    model = get_relu_model()
    compiled_model = core.compile_model(model, device)
    input_name = "data"
    output_node = compiled_model.input(input_name)
    rt_info = output_node.rt_info
    assert isinstance(rt_info, RTMap)


def test_update_rt_info(device):
    relu = ops.relu(5)
    output_node = Output._from_node(relu)
    rt = output_node.get_rt_info()
    rt["test12345"] = "test"
    for key, value in output_node.get_rt_info().items():
        assert key == "test12345"
        assert isinstance(value, OVAny)


def test_operations():
    data = ops.parameter([2])
    split = ops.split(data, 0, 2)
    outputs = split.outputs()
    assert outputs[0] < outputs[1]
    assert outputs[0] == split.output(0)
    assert hash(outputs[0]) == hash(split.output(0))
    assert hash(outputs[0]) != hash(outputs[0].node)


def test_copy():
    node = ops.relu(5)
    output_node = node.outputs()[0]
    out_copy = copy(output_node)
    assert out_copy is not output_node
    assert out_copy == output_node
    assert out_copy.get_node() is output_node.get_node()
    out_copy._add_new_var = True
    output_node._add_new_var = False
    assert out_copy._add_new_var != output_node._add_new_var


def test_deepcopy():
    node = ops.relu(5)
    output_node = node.outputs()[0]
    with pytest.raises(TypeError) as e:
        deepcopy(output_node)
    assert "Cannot deepcopy 'openvino.Output' object." in str(e)
