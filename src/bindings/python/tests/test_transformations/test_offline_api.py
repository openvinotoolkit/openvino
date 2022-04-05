# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import numpy as np
from openvino.runtime import serialize
from openvino.offline_transformations import apply_moc_transformations, apply_pot_transformations, \
    apply_low_latency_transformation, apply_pruning_transformation, apply_make_stateful_transformation, \
    compress_model_transformation

from openvino.runtime import Model, PartialShape, Core
import openvino.runtime as ov


def get_test_function():
    param = ov.opset8.parameter(PartialShape([1, 3, 22, 22]), name="parameter")
    param.get_output_tensor(0).set_names({"parameter"})
    relu = ov.opset8.relu(param)
    res = ov.opset8.result(relu, name="result")
    res.get_output_tensor(0).set_names({"result"})
    return Model([res], [param], "test")


def test_moc_transformations():
    function = get_test_function()

    apply_moc_transformations(function, False)

    assert function is not None
    assert len(function.get_ops()) == 3


def test_pot_transformations():
    function = get_test_function()

    apply_pot_transformations(function, "GNA")

    assert function is not None
    assert len(function.get_ops()) == 3


def test_low_latency_transformation():
    function = get_test_function()

    apply_low_latency_transformation(function, True)

    assert function is not None
    assert len(function.get_ops()) == 3


def test_pruning_transformation():
    function = get_test_function()

    apply_pruning_transformation(function)

    assert function is not None
    assert len(function.get_ops()) == 3


def test_make_stateful_transformations():
    function = get_test_function()

    apply_make_stateful_transformation(function, {"parameter": "result"})

    assert function is not None
    assert len(function.get_parameters()) == 0
    assert len(function.get_results()) == 0


def test_serialize_pass_v2():
    core = Core()
    xml_path = "./serialized_function.xml"
    bin_path = "./serialized_function.bin"
    shape = [100, 100, 2]
    parameter_a = ov.opset8.parameter(shape, dtype=np.float32, name="A")
    parameter_b = ov.opset8.parameter(shape, dtype=np.float32, name="B")
    model = ov.opset8.floor(ov.opset8.minimum(ov.opset8.abs(parameter_a), parameter_b))
    func = Model(model, [parameter_a, parameter_b], "Model")

    serialize(func, xml_path, bin_path)

    assert func is not None

    res_func = core.read_model(model=xml_path, weights=bin_path)

    assert func.get_parameters() == res_func.get_parameters()
    assert func.get_ordered_ops() == res_func.get_ordered_ops()

    os.remove(xml_path)
    os.remove(bin_path)


def test_compress_model_transformation():
    node_constant = ov.opset8.constant(np.array([[0.0, 0.1, -0.1], [-2.5, 2.5, 3.0]], dtype=np.float32))
    node_ceil = ov.opset8.ceiling(node_constant)
    func = Model(node_ceil, [], "TestFunction")
    assert func.get_ordered_ops()[0].get_element_type().get_type_name() == "f32"
    compress_model_transformation(func)

    assert func is not None
    assert func.get_ordered_ops()[0].get_element_type().get_type_name() == "f16"


def test_Version_default():
    core = Core()
    xml_path = "./serialized_function.xml"
    bin_path = "./serialized_function.bin"
    shape = [100, 100, 2]
    parameter_a = ov.opset8.parameter(shape, dtype=np.float32, name="A")
    parameter_b = ov.opset8.parameter(shape, dtype=np.float32, name="B")
    model = ov.opset8.floor(ov.opset8.minimum(ov.opset8.abs(parameter_a), parameter_b))
    func = Model(model, [parameter_a, parameter_b], "Model")

    serialize(func, xml_path, bin_path)
    res_func = core.read_model(model=xml_path, weights=bin_path)

    assert func.get_parameters() == res_func.get_parameters()
    assert func.get_ordered_ops() == res_func.get_ordered_ops()

    os.remove(xml_path)
    os.remove(bin_path)


def test_serialize_default_bin():
    xml_path = "./serialized_function.xml"
    bin_path = "./serialized_function.bin"
    model = get_test_function()
    serialize(model, xml_path)
    assert os.path.exists(bin_path)
    os.remove(xml_path)
    os.remove(bin_path)


def test_Version_ir_v10():
    core = Core()
    xml_path = "./serialized_function.xml"
    bin_path = "./serialized_function.bin"
    shape = [100, 100, 2]
    parameter_a = ov.opset8.parameter(shape, dtype=np.float32, name="A")
    parameter_b = ov.opset8.parameter(shape, dtype=np.float32, name="B")
    model = ov.opset8.floor(ov.opset8.minimum(ov.opset8.abs(parameter_a), parameter_b))
    func = Model(model, [parameter_a, parameter_b], "Model")

    serialize(func, xml_path, bin_path, "IR_V10")
    res_func = core.read_model(model=xml_path, weights=bin_path)

    assert func.get_parameters() == res_func.get_parameters()
    assert func.get_ordered_ops() == res_func.get_ordered_ops()

    os.remove(xml_path)
    os.remove(bin_path)


def test_Version_ir_v11():
    core = Core()
    xml_path = "./serialized_function.xml"
    bin_path = "./serialized_function.bin"
    shape = [100, 100, 2]
    parameter_a = ov.opset8.parameter(shape, dtype=np.float32, name="A")
    parameter_b = ov.opset8.parameter(shape, dtype=np.float32, name="B")
    model = ov.opset8.floor(ov.opset8.minimum(ov.opset8.abs(parameter_a), parameter_b))
    func = Model(model, [parameter_a, parameter_b], "Model")

    serialize(func, xml_path, bin_path, "IR_V11")
    res_func = core.read_model(model=xml_path, weights=bin_path)

    assert func.get_parameters() == res_func.get_parameters()
    assert func.get_ordered_ops() == res_func.get_ordered_ops()

    os.remove(xml_path)
    os.remove(bin_path)
