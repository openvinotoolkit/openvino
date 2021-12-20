# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

from ..conftest import model_path
from openvino.runtime import ConstOutput, Shape, PartialShape, Type

from openvino.runtime import Core

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
    exptected_string = "openvino.runtime.ConstOutput wraps ov::Output<Const ov::Node >"
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
