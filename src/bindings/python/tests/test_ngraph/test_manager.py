# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa

import json
import os

import numpy as np
import pytest

import openvino.runtime.opset8 as ov
from openvino.runtime import Model, PartialShape, Shape
from openvino.runtime.passes import Manager
from tests.test_ngraph.util import count_ops_of_type
from openvino.runtime import Core


def test_constant_folding():
    node_constant = ov.constant(np.array([[0.0, 0.1, -0.1], [-2.5, 2.5, 3.0]], dtype=np.float32))
    node_ceil = ov.ceiling(node_constant)
    func = Model(node_ceil, [], "TestFunction")

    assert count_ops_of_type(func, node_ceil) == 1
    assert count_ops_of_type(func, node_constant) == 1

    pass_manager = Manager()
    pass_manager.register_pass("ConstantFolding")
    pass_manager.run_passes(func)

    assert count_ops_of_type(func, node_ceil) == 0
    assert count_ops_of_type(func, node_constant) == 1

    new_const = func.get_results()[0].input(0).get_source_output().get_node()
    values_out = new_const.get_vector()
    values_expected = [0.0, 1.0, 0.0, -2.0, 3.0, 3.0]

    assert np.allclose(values_out, values_expected)


def test_serialize_seperate_paths_kwargs():
    core = Core()
    xml_path = "serialized_function.xml"
    bin_path = "serialized_function.bin"
    shape = [2, 2]
    parameter_a = ov.parameter(shape, dtype=np.float32, name="A")
    parameter_b = ov.parameter(shape, dtype=np.float32, name="B")
    parameter_c = ov.parameter(shape, dtype=np.float32, name="C")
    model = (parameter_a + parameter_b) * parameter_c
    func = Model(model, [parameter_a, parameter_b, parameter_c], "Model")

    pass_manager = Manager()
    pass_manager.register_pass(pass_name="Serialize", xml_path=xml_path, bin_path=bin_path)
    pass_manager.run_passes(func)

    res_func = core.read_model(model=xml_path, weights=bin_path)

    assert func.get_parameters() == res_func.get_parameters()
    assert func.get_ordered_ops() == res_func.get_ordered_ops()

    os.remove(xml_path)
    os.remove(bin_path)

def test_serialize_seperate_paths_args():
    core = Core()
    xml_path = "serialized_function.xml"
    bin_path = "serialized_function.bin"
    shape = [2, 2]
    parameter_a = ov.parameter(shape, dtype=np.float32, name="A")
    parameter_b = ov.parameter(shape, dtype=np.float32, name="B")
    parameter_c = ov.parameter(shape, dtype=np.float32, name="C")
    parameter_d = ov.parameter(shape, dtype=np.float32, name="D")
    model = ((parameter_a + parameter_b) * parameter_c) / parameter_d
    func = Model(model, [parameter_a, parameter_b, parameter_c, parameter_d], "Model")

    pass_manager = Manager()
    pass_manager.register_pass("Serialize", xml_path, bin_path)
    pass_manager.run_passes(func)

    res_func = core.read_model(model=xml_path, weights=bin_path)

    assert func.get_parameters() == res_func.get_parameters()
    assert func.get_ordered_ops() == res_func.get_ordered_ops()

    os.remove(xml_path)
    os.remove(bin_path)


def test_serialize_pass_mixed_args_kwargs():
    core = Core()
    xml_path = "serialized_function.xml"
    bin_path = "serialized_function.bin"
    shape = [3, 2]
    parameter_a = ov.parameter(shape, dtype=np.float32, name="A")
    parameter_b = ov.parameter(shape, dtype=np.float32, name="B")
    model = parameter_a - parameter_b
    func = Model(model, [parameter_a, parameter_b], "Model")

    pass_manager = Manager()
    pass_manager.register_pass("Serialize", xml_path, bin_path=bin_path)
    pass_manager.run_passes(func)

    res_func = core.read_model(model=xml_path, weights=bin_path)

    assert func.get_parameters() == res_func.get_parameters()
    assert func.get_ordered_ops() == res_func.get_ordered_ops()

    os.remove(xml_path)
    os.remove(bin_path)


def test_serialize_pass_mixed_args_kwargs_v2():
    core = Core()
    xml_path = "./serialized_function.xml"
    bin_path = "./serialized_function.bin"
    shape = [100, 100, 2]
    parameter_a = ov.parameter(shape, dtype=np.float32, name="A")
    parameter_b = ov.parameter(shape, dtype=np.float32, name="B")
    parameter_c = ov.parameter(shape, dtype=np.float32, name="C")
    parameter_d = ov.parameter(shape, dtype=np.float32, name="D")
    model = ov.floor(ov.minimum(ov.abs(parameter_a), ov.multiply(parameter_b, parameter_c)))
    func = Model(model, [parameter_a, parameter_b, parameter_c], "Model")
    pass_manager = Manager()
    pass_manager.register_pass("Serialize", xml_path=xml_path, bin_path=bin_path)
    pass_manager.run_passes(func)

    res_func = core.read_model(model=xml_path, weights=bin_path)

    assert func.get_parameters() == res_func.get_parameters()
    assert func.get_ordered_ops() == res_func.get_ordered_ops()

    os.remove(xml_path)
    os.remove(bin_path)


def test_serialize_pass_wrong_num_of_args():
    xml_path = "serialized_function.xml"
    bin_path = "serialized_function.bin"

    pass_manager = Manager()
    with pytest.raises(TypeError) as e:
        pass_manager.register_pass(pass_name="Serialize", xml_path=xml_path, bin_path=bin_path, model=5)
    assert "Invoked with:" in str(e.value)


def test_serialize_results():
    core = Core()
    node_constant = ov.constant(np.array([[0.0, 0.1, -0.1], [-2.5, 2.5, 3.0]], dtype=np.float32))
    node_ceil = ov.ceiling(node_constant)
    func = Model(node_ceil, [], "Model")

    xml_path = "serialized_function.xml"
    bin_path = "serialized_function.bin"
    pass_manager = Manager()
    pass_manager.register_pass("Serialize", xml_path=xml_path, bin_path=bin_path)
    pass_manager.run_passes(func)

    res_func = core.read_model(model=xml_path, weights=bin_path)
    const = func.get_results()[0].input(0).get_source_output().get_node()
    new_const = res_func.get_results()[0].input(0).get_source_output().get_node()

    assert const == new_const

    os.remove(xml_path)
    os.remove(bin_path)


def test_serialize_pass_tuple():
    core = Core()
    xml_path = "./serialized_function.xml"
    bin_path = "./serialized_function.bin"
    shape = [100, 100, 2]
    parameter_a = ov.parameter(shape, dtype=np.float32, name="A")
    parameter_b = ov.parameter(shape, dtype=np.float32, name="B")
    parameter_c = ov.parameter(shape, dtype=np.float32, name="C")
    parameter_d = ov.parameter(shape, dtype=np.float32, name="D")
    model = ov.floor(ov.minimum(ov.abs(parameter_a), ov.multiply(parameter_b, parameter_c)))
    func = Model(model, [parameter_a, parameter_b, parameter_c], "Model")
    pass_manager = Manager()
    pass_manager.register_pass("Serialize", output_files=(xml_path, bin_path))
    pass_manager.run_passes(func)

    res_func = core.read_model(model=xml_path, weights=bin_path)

    assert func.get_parameters() == res_func.get_parameters()
    assert func.get_ordered_ops() == res_func.get_ordered_ops()

    os.remove(xml_path)
    os.remove(bin_path)


def test_default_version():
    core = Core()
    xml_path = "./serialized_function.xml"
    bin_path = "./serialized_function.bin"
    shape = [100, 100, 2]
    parameter_a = ov.parameter(shape, dtype=np.float32, name="A")
    parameter_b = ov.parameter(shape, dtype=np.float32, name="B")
    parameter_c = ov.parameter(shape, dtype=np.float32, name="C")
    parameter_d = ov.parameter(shape, dtype=np.float32, name="D")
    model = ov.floor(ov.minimum(ov.abs(parameter_a), ov.multiply(parameter_b, parameter_c)))
    func = Model(model, [parameter_a, parameter_b, parameter_c], "Model")
    pass_manager = Manager()
    pass_manager.register_pass("Serialize", output_files=(xml_path, bin_path))
    pass_manager.run_passes(func)

    res_func = core.read_model(model=xml_path, weights=bin_path)

    assert func.get_parameters() == res_func.get_parameters()
    assert func.get_ordered_ops() == res_func.get_ordered_ops()

    os.remove(xml_path)
    os.remove(bin_path)


def test_default_version_IR_V11_tuple():
    core = Core()
    xml_path = "./serialized_function.xml"
    bin_path = "./serialized_function.bin"
    shape = [100, 100, 2]
    parameter_a = ov.parameter(shape, dtype=np.float32, name="A")
    parameter_b = ov.parameter(shape, dtype=np.float32, name="B")
    parameter_c = ov.parameter(shape, dtype=np.float32, name="C")
    parameter_d = ov.parameter(shape, dtype=np.float32, name="D")
    model = ov.floor(ov.minimum(ov.abs(parameter_a), ov.multiply(parameter_b, parameter_c)))
    func = Model(model, [parameter_a, parameter_b, parameter_c], "Model")
    pass_manager = Manager()
    pass_manager.register_pass("Serialize", output_files=(xml_path, bin_path), version="IR_V11")
    pass_manager.run_passes(func)

    res_func = core.read_model(model=xml_path, weights=bin_path)

    assert func.get_parameters() == res_func.get_parameters()
    assert func.get_ordered_ops() == res_func.get_ordered_ops()

    os.remove(xml_path)
    os.remove(bin_path)


def test_default_version_IR_V11_seperate_paths():
    core = Core()
    xml_path = "./serialized_function.xml"
    bin_path = "./serialized_function.bin"
    shape = [100, 100, 2]
    parameter_a = ov.parameter(shape, dtype=np.float32, name="A")
    parameter_b = ov.parameter(shape, dtype=np.float32, name="B")
    parameter_c = ov.parameter(shape, dtype=np.float32, name="C")
    parameter_d = ov.parameter(shape, dtype=np.float32, name="D")
    model = ov.floor(ov.minimum(ov.abs(parameter_a), ov.multiply(parameter_b, parameter_c)))
    func = Model(model, [parameter_a, parameter_b, parameter_c], "Model")
    pass_manager = Manager()
    pass_manager.register_pass("Serialize", xml_path=xml_path, bin_path=bin_path, version="IR_V11")
    pass_manager.run_passes(func)

    res_func = core.read_model(model=xml_path, weights=bin_path)

    assert func.get_parameters() == res_func.get_parameters()
    assert func.get_ordered_ops() == res_func.get_ordered_ops()

    os.remove(xml_path)
    os.remove(bin_path)
