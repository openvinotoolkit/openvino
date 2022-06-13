# -*- coding: utf-8 -*-
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

from ..conftest import model_path
import openvino.runtime.opset8 as ops
from openvino.runtime import (
    ConstOutput,
    Shape,
    PartialShape,
    Type,
    Output,
    RTMap,
    OVAny,
    Core,
)


is_myriad = os.environ.get("TEST_DEVICE") == "MYRIAD"
test_net_xml, test_net_bin = model_path(is_myriad)


def model_path(is_myriad=False):
    path_to_repo = os.environ["MODELS_PATH"]
    if not is_myriad:
        test_xml = os.path.join(path_to_repo, "models", "test_model", "test_model_fp32.xml")
        test_bin = os.path.join(path_to_repo, "models", "test_model", "test_model_fp32.bin")
    else:
        test_xml = os.path.join(path_to_repo, "models", "test_model", "test_model_fp16.xml")
        test_bin = os.path.join(path_to_repo, "models", "test_model", "test_model_fp16.bin")
    return (test_xml, test_bin)


def test_const_output_type(device):
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    node = exec_net.input(0)
    assert isinstance(node, ConstOutput)


def test_const_output_docs(device):
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    node = exec_net.input(0)
    exptected_string = "openvino.runtime.ConstOutput represents port/node output."
    assert node.__doc__ == exptected_string


def test_const_output_get_index(device):
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    node = exec_net.input("data")
    assert node.get_index() == 0
    assert node.index == 0


def test_const_output_get_element_type(device):
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    node = exec_net.input("data")
    assert node.get_element_type() == Type.f32
    assert node.element_type == Type.f32


def test_const_output_get_shape(device):
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    node = exec_net.input("data")
    expected_shape = Shape([1, 3, 32, 32])
    assert str(node.get_shape()) == str(expected_shape)
    assert str(node.shape) == str(expected_shape)


def test_const_output_get_partial_shape(device):
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    node = exec_net.input("data")
    expected_partial_shape = PartialShape([1, 3, 32, 32])
    assert node.get_partial_shape() == expected_partial_shape
    assert node.partial_shape == expected_partial_shape


def test_const_output_get_target_inputs(device):
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    outputs = exec_net.outputs
    for node in outputs:
        assert isinstance(node.get_target_inputs(), set)
        assert isinstance(node.target_inputs, set)


def test_const_output_get_names(device):
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    input_name = "data"
    node = exec_net.input(input_name)
    expected_names = set()
    expected_names.add(input_name)
    assert node.get_names() == expected_names
    assert node.names == expected_names
    assert node.get_any_name() == input_name
    assert node.any_name == input_name


def test_const_get_rf_info(device):
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    output_node = exec_net.output(0)
    rt_info = output_node.get_rt_info()
    assert isinstance(rt_info, RTMap)


def test_const_output_runtime_info(device):
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    input_name = "data"
    output_node = exec_net.input(input_name)
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
