# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

from ..conftest import model_path
from openvino.runtime import Input, Shape, PartialShape, Type, RTMap
from openvino._pyopenvino import DescriptorTensor
import openvino.runtime.opset8 as ops

from openvino.runtime import Core, OVAny

test_net_xml, test_net_bin = model_path()


def test_input_type(device):
    core = Core()
    model = core.read_model(model=test_net_xml, weights=test_net_bin)
    compiled_model = core.compile_model(model, device)
    net_input = compiled_model.output(0)
    input_node = net_input.get_node().inputs()[0]
    assert isinstance(input_node, Input)


def test_const_output_docs(device):
    core = Core()
    model = core.read_model(model=test_net_xml, weights=test_net_bin)
    compiled_model = core.compile_model(model, device)
    net_input = compiled_model.output(0)
    input_node = net_input.get_node().inputs()[0]
    exptected_string = "openvino.runtime.Input wraps ov::Input<Node>"
    assert input_node.__doc__ == exptected_string


def test_input_get_index(device):
    core = Core()
    model = core.read_model(model=test_net_xml, weights=test_net_bin)
    compiled_model = core.compile_model(model, device)
    net_input = compiled_model.output(0)
    input_node = net_input.get_node().inputs()[0]
    assert input_node.get_index() == 0


def test_input_element_type(device):
    core = Core()
    model = core.read_model(model=test_net_xml, weights=test_net_bin)
    compiled_model = core.compile_model(model, device)
    net_input = compiled_model.output(0)
    input_node = net_input.get_node().inputs()[0]
    assert input_node.get_element_type() == Type.f32


def test_input_get_shape(device):
    core = Core()
    model = core.read_model(model=test_net_xml, weights=test_net_bin)
    compiled_model = core.compile_model(model, device)
    net_input = compiled_model.output(0)
    input_node = net_input.get_node().inputs()[0]
    assert str(input_node.get_shape()) == str(Shape([1, 10]))


def test_input_get_partial_shape(device):
    core = Core()
    model = core.read_model(model=test_net_xml, weights=test_net_bin)
    compiled_model = core.compile_model(model, device)
    net_input = compiled_model.output(0)
    input_node = net_input.get_node().inputs()[0]
    expected_partial_shape = PartialShape([1, 10])
    assert input_node.get_partial_shape() == expected_partial_shape


def test_input_get_source_output(device):
    core = Core()
    model = core.read_model(model=test_net_xml, weights=test_net_bin)
    compiled_model = core.compile_model(model, device)
    net_input = compiled_model.output(0)
    input_node = net_input.get_node().inputs()[0]
    name = input_node.get_source_output().get_node().get_friendly_name()
    assert name == "fc_out"


def test_input_get_tensor(device):
    core = Core()
    model = core.read_model(model=test_net_xml, weights=test_net_bin)
    compiled_model = core.compile_model(model, device)
    net_input = compiled_model.output(0)
    input_node = net_input.get_node().inputs()[0]
    tensor = input_node.get_tensor()
    assert isinstance(tensor, DescriptorTensor)


def test_input_get_rt_info(device):
    core = Core()
    model = core.read_model(model=test_net_xml, weights=test_net_bin)
    compiled_model = core.compile_model(model, device)
    net_input = compiled_model.output(0)
    input_node = net_input.get_node().inputs()[0]
    rt_info = input_node.get_rt_info()
    assert isinstance(rt_info, RTMap)


def test_input_rt_info(device):
    core = Core()
    model = core.read_model(model=test_net_xml, weights=test_net_bin)
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
    model = core.read_model(model=test_net_xml, weights=test_net_bin)
    compiled_model = core.compile_model(model, device)
    net_input = compiled_model.output(0)
    input_node = net_input.get_node().inputs()[0]
    rt = input_node.get_rt_info()
    rt["test12345"] = "test"
    for key, value in input_node.get_rt_info().items():
        assert key == "test12345"
        assert isinstance(value, str)
