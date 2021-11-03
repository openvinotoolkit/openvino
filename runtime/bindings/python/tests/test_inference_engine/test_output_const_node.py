# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

from ..conftest import model_path
from openvino.impl import ConstOutput, Shape, PartialShape, Type

from openvino import Core

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
    exptected_string = "openvino.impl.ConstOutput wraps ov::Output<Const ov::Node >"
    assert node.__doc__ == exptected_string


def test_const_output_get_index(device):
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    node = exec_net.input("data")
    assert node.get_index() == 0


def test_const_output_get_element_type(device):
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    node = exec_net.input("data")
    assert node.get_element_type() == Type.f32


def test_const_output_get_shape(device):
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    node = exec_net.input("data")
    expected_shape = Shape([1, 3, 32, 32])
    assert str(node.get_shape()) == str(expected_shape)


def test_const_output_get_partial_shape(device):
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    node = exec_net.input("data")
    expected_partial_shape = PartialShape([1, 3, 32, 32])
    assert node.get_partial_shape() == expected_partial_shape


def test_const_output_get_target_inputs(device):
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    outputs = exec_net.outputs
    for node in outputs:
        assert isinstance(node.get_target_inputs(), set)
