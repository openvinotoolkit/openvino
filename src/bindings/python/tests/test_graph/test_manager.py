# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import pytest

import openvino.opset10 as ops
from openvino import Core, Model
from openvino.passes import Manager, Serialize, ConstantFolding, Version

from tests.test_graph.util import count_ops_of_type
from tests.utils.helpers import create_filenames_for_ir, compare_models


def create_model():
    shape = [100, 100, 2]
    parameter_a = ops.parameter(shape, dtype=np.float32, name="A")
    parameter_b = ops.parameter(shape, dtype=np.float32, name="B")
    parameter_c = ops.parameter(shape, dtype=np.float32, name="C")
    floor_op = ops.floor(ops.minimum(ops.abs(parameter_a), ops.multiply(parameter_b, parameter_c)))
    model = Model(floor_op, [parameter_a, parameter_b, parameter_c], "Model")
    return model


def test_constant_folding():
    node_constant = ops.constant(np.array([[0.0, 0.1, -0.1], [-2.5, 2.5, 3.0]], dtype=np.float32))
    node_ceil = ops.ceiling(node_constant)
    model = Model(node_ceil, [], "TestModel")

    assert count_ops_of_type(model, node_ceil) == 1
    assert count_ops_of_type(model, node_constant) == 1

    pass_manager = Manager()
    pass_manager.register_pass(ConstantFolding())
    pass_manager.run_passes(model)

    assert count_ops_of_type(model, node_ceil) == 0
    assert count_ops_of_type(model, node_constant) == 1

    result = model.get_results()[0]
    new_const = result.input(0).get_source_output().get_node()
    values_out = new_const.get_vector()
    values_expected = [0.0, 1.0, 0.0, -2.0, 3.0, 3.0]

    assert np.allclose(values_out, values_expected)


# request - https://docs.pytest.org/en/7.1.x/reference/reference.html#request
@pytest.fixture
def prepare_ir_paths(request, tmp_path):
    xml_path, bin_path = create_filenames_for_ir(request.node.name, tmp_path)

    yield xml_path, bin_path

    # IR Files deletion should be done after `Model` is destructed.
    # It may be achieved by splitting scopes (`Model` will be destructed
    # just after test scope finished), or by calling `del Model`
    os.remove(xml_path)
    os.remove(bin_path)


def test_serialize_separate_paths_kwargs(prepare_ir_paths):
    core = Core()

    shape = [2, 2]
    parameter_a = ops.parameter(shape, dtype=np.float32, name="A")
    parameter_b = ops.parameter(shape, dtype=np.float32, name="B")
    parameter_c = ops.parameter(shape, dtype=np.float32, name="C")
    _model = (parameter_a + parameter_b) * parameter_c
    model = Model(_model, [parameter_a, parameter_b, parameter_c], "Model")

    xml_path, bin_path = prepare_ir_paths
    pass_manager = Manager()
    pass_manager.register_pass(Serialize(path_to_xml=xml_path, path_to_bin=bin_path))
    pass_manager.run_passes(model)

    res_model = core.read_model(model=xml_path, weights=bin_path)

    assert compare_models(model, res_model)


def test_serialize_separate_paths_args(prepare_ir_paths):
    core = Core()

    shape = [2, 2]
    parameter_a = ops.parameter(shape, dtype=np.float32, name="A")
    parameter_b = ops.parameter(shape, dtype=np.float32, name="B")
    parameter_c = ops.parameter(shape, dtype=np.float32, name="C")
    parameter_d = ops.parameter(shape, dtype=np.float32, name="D")
    _model = ((parameter_a + parameter_b) * parameter_c) / parameter_d
    model = Model(_model, [parameter_a, parameter_b, parameter_c, parameter_d], "Model")

    xml_path, bin_path = prepare_ir_paths
    pass_manager = Manager()
    pass_manager.register_pass(Serialize(xml_path, bin_path))
    pass_manager.run_passes(model)

    res_model = core.read_model(model=xml_path, weights=bin_path)

    assert compare_models(model, res_model)


def test_serialize_pass_mixed_args_kwargs(prepare_ir_paths):
    core = Core()

    shape = [3, 2]
    parameter_a = ops.parameter(shape, dtype=np.float32, name="A")
    parameter_b = ops.parameter(shape, dtype=np.float32, name="B")
    _model = parameter_a - parameter_b
    model = Model(_model, [parameter_a, parameter_b], "Model")

    xml_path, bin_path = prepare_ir_paths
    pass_manager = Manager()
    pass_manager.register_pass(Serialize(xml_path, path_to_bin=bin_path))
    pass_manager.run_passes(model)

    res_model = core.read_model(model=xml_path, weights=bin_path)

    assert compare_models(model, res_model)


def test_serialize_pass_mixed_args_kwargs_v2(prepare_ir_paths):
    core = Core()

    xml_path, bin_path = prepare_ir_paths
    model = create_model()
    pass_manager = Manager()
    pass_manager.register_pass(Serialize(path_to_xml=xml_path, path_to_bin=bin_path))
    pass_manager.run_passes(model)

    res_model = core.read_model(model=xml_path, weights=bin_path)

    assert compare_models(model, res_model)


# request - https://docs.pytest.org/en/7.1.x/reference/reference.html#request
def test_serialize_pass_wrong_num_of_args(request, tmp_path):
    xml_path, bin_path = create_filenames_for_ir(request.node.name, tmp_path)

    pass_manager = Manager()
    with pytest.raises(TypeError) as e:
        pass_manager.register_pass(Serialize(path_to_xml=xml_path, path_to_bin=bin_path, model=5))
    assert "Invoked with:" in str(e.value)


def test_serialize_results(prepare_ir_paths):
    core = Core()
    node_constant = ops.constant(np.array([[0.0, 0.1, -0.1], [-2.5, 2.5, 3.0]], dtype=np.float32))
    node_ceil = ops.ceiling(node_constant)
    model = Model(node_ceil, [], "Model")

    xml_path, bin_path = prepare_ir_paths
    pass_manager = Manager()
    pass_manager.register_pass(Serialize(path_to_xml=xml_path, path_to_bin=bin_path))
    pass_manager.run_passes(model)

    res_model = core.read_model(model=xml_path, weights=bin_path)

    assert compare_models(model, res_model)


def test_default_version(prepare_ir_paths):
    core = Core()

    xml_path, bin_path = prepare_ir_paths
    model = create_model()
    pass_manager = Manager()
    pass_manager.register_pass(Serialize(xml_path, bin_path))
    pass_manager.run_passes(model)

    res_model = core.read_model(model=xml_path, weights=bin_path)

    assert compare_models(model, res_model)


def test_default_version_ir_v11_separate_paths(prepare_ir_paths):
    core = Core()

    xml_path, bin_path = prepare_ir_paths
    model = create_model()
    pass_manager = Manager()
    pass_manager.register_pass(Serialize(path_to_xml=xml_path, path_to_bin=bin_path, version=Version.IR_V11))
    pass_manager.run_passes(model)

    res_model = core.read_model(model=xml_path, weights=bin_path)

    assert compare_models(model, res_model)
