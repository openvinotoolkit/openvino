# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest

import openvino.opset8 as ops
import openvino as ov

from openvino import (
    Model,
    Layout,
    PartialShape,
    Shape,
    Strides,
    Tensor,
    Type,
    layout_helpers,
)

from openvino.op import Parameter, Constant
from openvino.op.util import VariableInfo, Variable
from openvino import AxisVector, Coordinate, CoordinateDiff
from openvino._pyopenvino import DescriptorTensor

from openvino.utils.types import get_element_type
from tests.utils.helpers import generate_model_with_memory


def test_graph_api():
    shape = [2, 2]
    parameter_a = ops.parameter(shape, dtype=np.float32, name="A")
    parameter_b = ops.parameter(shape, dtype=Type.f32, name="B")
    parameter_c = ops.parameter(shape, dtype=np.float32, name="C")
    model = (parameter_a + parameter_b) * parameter_c

    assert parameter_a.element_type == Type.f32
    assert parameter_b.element_type == Type.f32
    assert parameter_a.partial_shape == PartialShape([2, 2])
    parameter_a.layout = Layout("NC")
    assert parameter_a.layout == Layout("NC")
    model = Model(model, [parameter_a, parameter_b, parameter_c], "TestModel")

    model.get_parameters()[1].set_partial_shape(PartialShape([3, 4, 5]))

    ordered_ops = model.get_ordered_ops()
    op_types = [op.get_type_name() for op in ordered_ops]
    assert op_types == ["Parameter", "Parameter", "Parameter", "Add", "Multiply", "Result"]
    assert len(model.get_ops()) == 6
    assert model.get_output_size() == 1
    assert ["A", "B", "C"] == [input.get_node().friendly_name for input in model.inputs]
    assert ["Result"] == [output.get_node().get_type_name() for output in model.outputs]
    assert model.input(0).get_node().friendly_name == "A"
    assert model.output(0).get_node().get_type_name() == "Result"
    assert model.input(tensor_name="A").get_node().friendly_name == "A"
    assert model.output().get_node().get_type_name() == "Result"
    assert model.get_output_op(0).get_type_name() == "Result"
    assert model.get_output_element_type(0) == parameter_a.get_element_type()
    assert list(model.get_output_shape(0)) == [2, 2]
    assert (model.get_parameters()[1].get_partial_shape()) == PartialShape([3, 4, 5])
    assert len(model.get_parameters()) == 3
    results = model.get_results()
    assert len(results) == 1
    assert results[0].get_output_element_type(0) == Type.f32
    assert results[0].get_output_partial_shape(0) == PartialShape([2, 2])
    results[0].layout = Layout("NC")
    assert results[0].layout.to_string() == Layout("NC")
    assert model.get_friendly_name() == "TestModel"


@pytest.mark.parametrize(
    "dtype",
    [
        np.float32,
        np.float64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        Type.f16,
        Type.f32,
        Type.f64,
        Type.i8,
        Type.i16,
        Type.i32,
        Type.i64,
        Type.u8,
        Type.u16,
        Type.u32,
        Type.u64,
    ],
)
def test_simple_model_on_parameters(dtype):
    shape = [2, 2]
    parameter_a = ops.parameter(shape, dtype=dtype, name="A")
    parameter_b = ops.parameter(shape, dtype=dtype, name="B")
    parameter_c = ops.parameter(shape, dtype=dtype, name="C")
    model = (parameter_a + parameter_b) * parameter_c
    expected_type = dtype if isinstance(dtype, Type) else get_element_type(dtype)
    assert model.get_type_name() == "Multiply"
    assert model.get_output_size() == 1
    assert model.get_output_element_type(0) == expected_type
    assert list(model.get_output_shape(0)) == [2, 2]


@pytest.mark.parametrize(
    ("input_shape", "dtype", "new_shape", "axis_mapping", "mode"),
    [
        ((3,), np.int32, [3, 3], [], []),
        ((4,), np.float32, [3, 4, 2, 4], [], []),
        ((3,), np.int8, [3, 3], [[0]], ["EXPLICIT"]),
    ],
)
def test_broadcast(input_shape, dtype, new_shape, axis_mapping, mode):
    input_data = ops.parameter(input_shape, name="input_data", dtype=dtype)
    node = ops.broadcast(input_data, new_shape, *axis_mapping, *mode)
    assert node.get_type_name() == "Broadcast"
    assert node.get_output_size() == 1
    assert node.get_output_element_type(0) == get_element_type(dtype)
    assert list(node.get_output_shape(0)) == new_shape


@pytest.mark.parametrize(
    ("destination_type", "input_data"),
    [(bool, np.zeros((2, 2), dtype=np.int32)), ("boolean", np.zeros((2, 2), dtype=np.int32))],
)
def test_convert_to_bool(destination_type, input_data):
    node = ops.convert(input_data, destination_type)
    assert node.get_type_name() == "Convert"
    assert node.get_output_size() == 1
    assert node.get_output_element_type(0) == Type.boolean
    assert list(node.get_output_shape(0)) == [2, 2]


@pytest.mark.parametrize(
    ("destination_type", "rand_range", "in_dtype", "expected_type"),
    [
        pytest.param(np.float32, (-8, 8), np.int32, np.float32),
        pytest.param(np.float64, (-16383, 16383), np.int64, np.float64),
        pytest.param("f32", (-8, 8), np.int32, np.float32),
        pytest.param("f64", (-16383, 16383), np.int64, np.float64),
        pytest.param(Type.f32, (-8, 8), np.int32, np.float32),
        pytest.param(Type.f64, (-16383, 16383), np.int64, np.float64),
    ],
)
def test_convert_to_float(destination_type, rand_range, in_dtype, expected_type):
    np.random.seed(133391)
    input_data = np.random.randint(*rand_range, size=(2, 2), dtype=in_dtype)
    node = ops.convert(input_data, destination_type)
    assert node.get_type_name() == "Convert"
    assert node.get_output_size() == 1
    assert node.get_output_element_type(0) == get_element_type(expected_type)
    assert list(node.get_output_shape(0)) == [2, 2]


@pytest.mark.parametrize(
    ("destination_type", "expected_type"),
    [
        (np.int8, np.int8),
        (np.int16, np.int16),
        (np.int32, np.int32),
        (np.int64, np.int64),
        ("i8", np.int8),
        ("i16", np.int16),
        ("i32", np.int32),
        ("i64", np.int64),
    ],
)
def test_convert_to_int(destination_type, expected_type):
    np.random.seed(133391)
    random_data = np.random.rand(2, 3, 4) * 16
    input_data = (np.ceil(-8 + random_data)).astype(expected_type)
    node = ops.convert(input_data, destination_type)
    assert node.get_type_name() == "Convert"
    assert node.get_output_size() == 1
    assert node.get_output_element_type(0) == get_element_type(expected_type)
    assert list(node.get_output_shape(0)) == [2, 3, 4]


@pytest.mark.parametrize(
    ("destination_type", "expected_type"),
    [
        (np.uint8, np.uint8),
        (np.uint16, np.uint16),
        (np.uint32, np.uint32),
        (np.uint64, np.uint64),
        ("u8", np.uint8),
        ("u16", np.uint16),
        ("u32", np.uint32),
        ("u64", np.uint64),
    ],
)
def test_convert_to_uint(destination_type, expected_type):
    np.random.seed(133391)
    input_data = np.ceil(np.random.rand(2, 3, 4) * 16).astype(expected_type)
    node = ops.convert(input_data, destination_type)
    assert node.get_type_name() == "Convert"
    assert node.get_output_size() == 1
    assert node.get_output_element_type(0) == get_element_type(expected_type)
    assert list(node.get_output_shape(0)) == [2, 3, 4]


def test_constant_get_data_bool():
    input_data = np.array([True, False, False, True])
    node = ops.constant(input_data, dtype=bool)
    retrieved_data = node.get_data()
    assert np.allclose(input_data, retrieved_data)


@pytest.mark.parametrize("data_type", [np.float32, np.float64])
def test_constant_get_data_floating_point(data_type):
    np.random.seed(133391)
    input_data = np.random.randn(2, 3, 4).astype(data_type)
    min_value = -1.0e20
    max_value = 1.0e20
    input_data = min_value + input_data * max_value * data_type(2)
    node = ops.constant(input_data, dtype=data_type)
    retrieved_data = node.get_data()
    assert np.allclose(input_data, retrieved_data)


@pytest.mark.parametrize("data_type", [np.int64, np.int32, np.int16, np.int8])
def test_constant_get_data_signed_integer(data_type):
    np.random.seed(133391)
    input_data = np.random.randint(
        np.iinfo(data_type).min, np.iinfo(data_type).max, size=[2, 3, 4], dtype=data_type,
    )
    node = ops.constant(input_data, dtype=data_type)
    retrieved_data = node.get_data()
    assert np.allclose(input_data, retrieved_data)


@pytest.mark.parametrize("data_type", [np.uint64, np.uint32, np.uint16, np.uint8])
def test_constant_get_data_unsigned_integer(data_type):
    np.random.seed(133391)
    input_data = np.random.randn(2, 3, 4).astype(data_type)
    input_data = (
        np.iinfo(data_type).min + input_data * np.iinfo(data_type).max + input_data * np.iinfo(data_type).max
    )
    node = ops.constant(input_data, dtype=data_type)
    retrieved_data = node.get_data()
    assert np.allclose(input_data, retrieved_data)


@pytest.mark.parametrize(
    "shared_flag",
    [
        (True),
        (False),
    ],
)
@pytest.mark.parametrize(
    "init_value",
    [
        (np.array([])),
        (np.array([], dtype=np.int32)),
        (np.empty(shape=(0))),
    ],
)
def test_constant_from_empty_array(shared_flag, init_value):
    const = Constant(init_value, shared_memory=shared_flag)
    assert tuple(const.shape) == init_value.shape
    assert const.get_element_type().to_dtype() == init_value.dtype
    assert const.get_byte_size() == init_value.nbytes
    assert np.allclose(const.data, init_value)


def test_set_argument():
    data1 = np.array([1, 2, 3])
    data2 = np.array([4, 5, 6])
    data3 = np.array([7, 8, 9])

    node1 = ops.constant(data1, dtype=np.float32)
    node2 = ops.constant(data2, dtype=np.float32)
    node3 = ops.constant(data3, dtype=np.float64)
    node4 = ops.constant(data3, dtype=np.float64)
    node_add = ops.add(node1, node2)

    # Original arguments
    node_inputs = node_add.inputs()
    assert node_inputs[0].get_element_type() == Type.f32
    assert node_inputs[1].get_element_type() == Type.f32
    assert len(node_inputs) == 2

    # Arguments changed by set_argument
    node_add.set_argument(0, node3.output(0))
    node_add.set_argument(1, node4.output(0))
    node_inputs = node_add.inputs()
    assert node_inputs[0].get_element_type() == Type.f64
    assert node_inputs[1].get_element_type() == Type.f64
    assert len(node_inputs) == 2

    # Arguments changed by set_argument(OutputVector)
    node_add.set_arguments([node1.output(0), node2.output(0)])
    assert node_inputs[0].get_element_type() == Type.f32
    assert node_inputs[1].get_element_type() == Type.f32
    assert len(node_inputs) == 2

    # Arguments changed by set_arguments(NodeVector)
    node_add.set_arguments([node3, node4])
    assert node_inputs[0].get_element_type() == Type.f64
    assert node_inputs[1].get_element_type() == Type.f64
    assert len(node_inputs) == 2


def test_clone_model():
    from copy import deepcopy
    # Create an original model
    shape = [2, 2]
    parameter_a = ops.parameter(shape, dtype=np.float32, name="A")
    parameter_b = ops.parameter(shape, dtype=np.float32, name="B")
    model_original = Model(parameter_a + parameter_b, [parameter_a, parameter_b])
    assert isinstance(model_original, Model)

    # Make copies of it
    model_copy2 = model_original.clone()
    model_copy3 = deepcopy(model_original)

    assert isinstance(model_copy2, Model)
    assert isinstance(model_copy3, Model)

    # Make changes to the copied models' inputs
    model_copy2.reshape({"A": [3, 3], "B": [3, 3]})
    model_copy3.reshape({"A": [3, 3], "B": [3, 3]})

    original_model_shapes = [single_input.get_shape() for single_input in model_original.inputs]
    model_copy2_shapes = [single_input.get_shape() for single_input in model_copy2.inputs]
    model_copy3_shapes = [single_input.get_shape() for single_input in model_copy3.inputs]

    assert original_model_shapes != model_copy2_shapes
    assert original_model_shapes != model_copy3_shapes
    assert model_copy2_shapes == model_copy3_shapes


def test_result():
    input_data = np.array([[11, 10], [1, 8], [3, 4]], dtype=np.float32)
    node = ops.result(input_data)
    assert node.get_type_name() == "Result"
    assert node.get_output_size() == 1
    assert node.get_output_element_type(0) == Type.f32
    assert list(node.get_output_shape(0)) == [3, 2]


def test_node_friendly_name():
    dummy_node = ops.parameter(shape=[1], name="dummy_name")
    assert (dummy_node.friendly_name == "dummy_name")

    dummy_node.set_friendly_name("changed_name")
    assert (dummy_node.get_friendly_name() == "changed_name")

    dummy_node.friendly_name = "new_name"
    assert (dummy_node.get_friendly_name() == "new_name")


def test_node_output():
    input_array = np.array([0, 1, 2, 3, 4, 5])
    splits = 3
    expected_shape = len(input_array) // splits

    input_tensor = ops.constant(input_array, dtype=np.int32)
    axis = ops.constant(0, dtype=np.int64)
    split_node = ops.split(input_tensor, axis, splits)

    split_node_outputs = split_node.outputs()

    assert len(split_node_outputs) == splits
    assert [output_node.get_index() for output_node in split_node_outputs] == [0, 1, 2]
    assert np.equal(
        [output_node.get_element_type() for output_node in split_node_outputs],
        input_tensor.get_element_type(),
    ).all()
    assert np.equal(
        [output_node.get_shape() for output_node in split_node_outputs],
        Shape([expected_shape]),
    ).all()
    assert np.equal(
        [output_node.get_partial_shape() for output_node in split_node_outputs],
        PartialShape([expected_shape]),
    ).all()

    output0 = split_node.output(0)
    output1 = split_node.output(1)
    output2 = split_node.output(2)

    assert [output0.get_index(), output1.get_index(), output2.get_index()] == [0, 1, 2]


def test_node_input_values():
    shapes = [Shape([3]), Shape([3])]
    data1 = np.array([1, 2, 3], dtype=np.int64)
    data2 = np.array([3, 2, 1], dtype=np.int64)

    node = ops.add(data1, data2)

    assert node.get_input_size() == 2
    assert node.get_input_element_type(0) == Type.i64
    assert node.get_input_partial_shape(0) == PartialShape([3])
    assert node.get_input_shape(1) == Shape([3])

    assert np.equal([input_node.get_shape() for input_node in node.input_values()], shapes,).all()
    assert np.equal([node.input_value(i).get_shape() for i in range(node.get_input_size())], shapes,).all()
    assert np.allclose(
        [input_node.get_node().get_vector() for input_node in node.input_values()],
        [data1, data2],
    )

    assert np.allclose(
        [node.input_value(i).get_node().get_vector() for i in range(node.get_input_size())],
        [data1, data2],
    )


def test_node_input_tensor():
    data1 = np.array([[1, 2, 3], [1, 2, 3]])
    data2 = np.array([3, 2, 1])

    node = ops.add(data1, data2)

    input_tensor1 = node.get_input_tensor(0)
    input_tensor2 = node.get_input_tensor(1)

    assert (isinstance(input_tensor1, DescriptorTensor))
    assert (isinstance(input_tensor2, DescriptorTensor))
    assert np.equal(input_tensor1.get_shape(), data1.shape).all()
    assert np.equal(input_tensor2.get_shape(), data2.shape).all()


def test_node_evaluate():
    data1 = np.array([3, 2, 3])
    data2 = np.array([4, 2, 3])
    expected_result = data1 + data2

    data1 = np.ascontiguousarray(data1)
    data2 = np.ascontiguousarray(data2)

    output = np.array([0, 0, 0])
    output = np.ascontiguousarray(output)

    node = ops.add(data1, data2)

    input_tensor1 = Tensor(array=data1, shared_memory=True)
    input_tensor2 = Tensor(array=data2, shared_memory=True)
    inputs_tensor_vector = [input_tensor1, input_tensor2]

    output_tensor_vector = [Tensor(array=output, shared_memory=True)]
    assert node.evaluate(output_tensor_vector, inputs_tensor_vector) is True
    assert np.equal(output_tensor_vector[0].data, expected_result).all()


def test_node_input():
    shape = [2, 2]
    parameter_a = ops.parameter(shape, dtype=np.float32, name="A")
    parameter_b = ops.parameter(shape, dtype=np.float32, name="B")

    model = parameter_a + parameter_b

    model_inputs = model.inputs()

    assert len(model_inputs) == 2
    assert [input_node.get_index() for input_node in model_inputs] == [0, 1]
    assert np.equal(
        [input_node.get_element_type() for input_node in model_inputs],
        model.get_element_type(),
    ).all()
    assert np.equal(
        [input_node.get_shape() for input_node in model_inputs], Shape(shape),
    ).all()
    assert np.equal(
        [input_node.get_partial_shape() for input_node in model_inputs],
        PartialShape(shape),
    ).all()

    input0 = model.input(0)
    input1 = model.input(1)

    assert [input0.get_index(), input1.get_index()] == [0, 1]


def test_node_target_inputs_soruce_output():
    shape = [2, 2]
    parameter_a = ops.parameter(shape, dtype=np.float32, name="A")
    parameter_b = ops.parameter(shape, dtype=np.float32, name="B")

    model = parameter_a + parameter_b

    out_a = list(parameter_a.output(0).get_target_inputs())[0]
    out_b = list(parameter_b.output(0).get_target_inputs())[0]

    assert out_a.get_node().name == model.name
    assert out_b.get_node().name == model.name
    assert np.equal([out_a.get_shape()], [model.get_output_shape(0)]).all()
    assert np.equal([out_b.get_shape()], [model.get_output_shape(0)]).all()

    in_model0 = model.input(0).get_source_output()
    in_model1 = model.input(1).get_source_output()

    assert in_model0.get_node().name == parameter_a.name
    assert in_model1.get_node().name == parameter_b.name
    assert np.equal([in_model0.get_shape()], [model.get_output_shape(0)]).all()
    assert np.equal([in_model1.get_shape()], [model.get_output_shape(0)]).all()


def test_runtime_info():
    test_shape = PartialShape([1, 1, 1, 1])
    test_type = Type.f32
    test_param = Parameter(test_type, test_shape)
    relu_node = ops.relu(test_param)
    runtime_info = relu_node.get_rt_info()
    runtime_info["affinity"] = "test_affinity"
    relu_node.set_friendly_name("testReLU")
    runtime_info_after = relu_node.get_rt_info()
    assert runtime_info_after["affinity"] == "test_affinity"


def test_multiple_outputs():
    input_shape = [4, 4]
    input_data = np.arange(-8, 8).reshape(input_shape).astype(np.float32)

    expected_output = np.split(input_data, 2, axis=1)[0]
    expected_output[expected_output < 0] = 0

    test_param = ops.parameter(input_shape, dtype=np.float32, name="A")
    split = ops.split(test_param, axis=1, num_splits=2)
    split_first_output = split.output(0)
    relu = ops.relu(split_first_output)

    assert relu.get_type_name() == "Relu"
    assert relu.get_output_size() == 1
    assert relu.get_output_element_type(0) == Type.f32
    assert list(relu.get_output_shape(0)) == [4, 2]


def test_sink_model_ctors():
    model = generate_model_with_memory(input_shape=[2, 2], data_type=np.float32)

    var_info = VariableInfo()
    var_info.data_shape = PartialShape([2, 1])
    var_info.data_type = Type.f32
    var_info.variable_id = "v1"
    variable_1 = Variable(var_info)

    init_val = ops.constant(np.zeros(Shape([2, 1])), Type.f32)

    rv = ops.read_value(init_val, variable_1)
    assign = ops.assign(rv, variable_1)

    model.add_variables([variable_1])
    model.add_sinks([assign])

    model.validate_nodes_and_infer_types()

    ordered_ops = model.get_ordered_ops()
    op_types = [op.get_type_name() for op in ordered_ops]
    sinks = model.get_sinks()
    assert ["Assign", "Assign"] == [sink.get_type_name() for sink in sinks]
    assert model.sinks[0].get_output_shape(0) == Shape([2, 2])
    assert op_types == ["Parameter", "Constant", "ReadValue", "Assign", "Constant", "ReadValue", "Add", "Assign", "Result"]
    assert len(model.get_ops()) == 9
    assert model.get_output_size() == 1
    assert model.get_output_op(0).get_type_name() == "Result"
    assert model.get_output_element_type(0) == model.get_parameters()[0].get_element_type()
    assert list(model.get_output_shape(0)) == [2, 2]
    assert (model.get_parameters()[0].get_partial_shape()) == PartialShape([2, 2])
    assert len(model.get_parameters()) == 1
    assert len(model.get_results()) == 1
    assert model.get_friendly_name() == "TestModel"


def test_sink_model_ctor_without_init_subgraph():
    input_data = ops.parameter([2, 2], name="input_data", dtype=np.float32)
    rv = ops.read_value("var_id_667", np.float32, [2, 2])
    add = ops.add(rv, input_data, name="MemoryAdd")
    node = ops.assign(add, "var_id_667")
    res = ops.result(add, "res")
    model = Model(results=[res], sinks=[node], parameters=[input_data], name="TestModel")

    var_info = VariableInfo()
    var_info.data_shape = PartialShape([2, 1])
    var_info.data_type = Type.f32
    var_info.variable_id = "v1"
    variable_1 = Variable(var_info)
    rv = ops.read_value(variable_1)
    assign = ops.assign(rv, variable_1)

    model.add_variables([variable_1])
    model.add_sinks([assign])

    ordered_ops = model.get_ordered_ops()
    op_types = [op.get_type_name() for op in ordered_ops]
    sinks = model.get_sinks()
    assert ["Assign", "Assign"] == [sink.get_type_name() for sink in sinks]
    assert model.sinks[0].get_output_shape(0) == Shape([2, 2])
    assert op_types == ["Parameter", "ReadValue", "Assign", "ReadValue", "Add", "Assign", "Result"]
    assert len(model.get_ops()) == 7
    assert model.get_output_size() == 1
    assert model.get_output_op(0).get_type_name() == "Result"
    assert model.get_output_element_type(0) == input_data.get_element_type()
    assert list(model.get_output_shape(0)) == [2, 2]
    assert (model.get_parameters()[0].get_partial_shape()) == PartialShape([2, 2])
    assert len(model.get_parameters()) == 1
    assert len(model.get_results()) == 1
    assert model.get_friendly_name() == "TestModel"


def test_strides_iteration_methods():
    data = np.array([1, 2, 3])
    strides = Strides(data)

    assert len(strides) == data.size
    assert np.equal(strides, data).all()
    assert np.equal([strides[i] for i in range(data.size)], data).all()

    data2 = np.array([5, 6, 7])
    for i in range(data2.size):
        strides[i] = data2[i]

    assert np.equal(strides, data2).all()


def test_axis_vector_iteration_methods():
    data = np.array([1, 2, 3])
    axis_vector = AxisVector(data)

    assert len(axis_vector) == data.size
    assert np.equal(axis_vector, data).all()
    assert np.equal([axis_vector[i] for i in range(data.size)], data).all()

    data2 = np.array([5, 6, 7])
    for i in range(data2.size):
        axis_vector[i] = data2[i]

    assert np.equal(axis_vector, data2).all()


def test_coordinate_iteration_methods():
    data = np.array([1, 2, 3])
    coordinate = Coordinate(data)

    assert len(coordinate) == data.size
    assert np.equal(coordinate, data).all()
    assert np.equal([coordinate[i] for i in range(data.size)], data).all()

    data2 = np.array([5, 6, 7])
    for i in range(data2.size):
        coordinate[i] = data2[i]

    assert np.equal(coordinate, data2).all()


def test_coordinate_diff_iteration_methods():
    data = np.array([1, 2, 3])
    coordinate_diff = CoordinateDiff(data)

    assert len(coordinate_diff) == data.size
    assert np.equal(coordinate_diff, data).all()
    assert np.equal([coordinate_diff[i] for i in range(data.size)], data).all()

    data2 = np.array([5, 6, 7])
    for i in range(data2.size):
        coordinate_diff[i] = data2[i]

    assert np.equal(coordinate_diff, data2).all()


def test_get_and_set_layout():
    shape = [2, 2]
    parameter_a = ops.parameter(shape, dtype=np.float32, name="A")
    parameter_b = ops.parameter(shape, dtype=np.float32, name="B")

    model = Model(parameter_a + parameter_b, [parameter_a, parameter_b])

    assert layout_helpers.get_layout(model.input(0)) == Layout()
    assert layout_helpers.get_layout(model.input(1)) == Layout()

    layout_helpers.set_layout(model.input(0), Layout("CH"))
    layout_helpers.set_layout(model.input(1), Layout("HW"))

    assert layout_helpers.get_layout(model.input(0)) == Layout("CH")
    assert layout_helpers.get_layout(model.input(1)) == Layout("HW")


def test_layout():
    layout = Layout("NCWH")
    layout2 = Layout("NCWH")
    scalar = Layout.scalar()
    scalar2 = Layout.scalar()

    assert layout == layout2
    assert layout != scalar
    assert scalar == scalar2
    assert scalar2 != layout2

    assert str(scalar) == str(scalar2)
    assert not (scalar.has_name("N"))
    assert not (scalar.has_name("C"))
    assert not (scalar.has_name("W"))
    assert not (scalar.has_name("H"))
    assert not (scalar.has_name("D"))

    assert layout.to_string() == layout2.to_string()
    assert layout.has_name("N")
    assert layout.has_name("C")
    assert layout.has_name("W")
    assert layout.has_name("H")
    assert not (layout.has_name("D"))
    assert layout.get_index_by_name("N") == 0
    assert layout.get_index_by_name("C") == 1
    assert layout.get_index_by_name("W") == 2
    assert layout.get_index_by_name("H") == 3

    layout = Layout("NC?")
    layout2 = Layout("N")
    assert layout != layout2
    assert str(layout) != str(layout2)
    assert layout.has_name("N")
    assert layout.has_name("C")
    assert not (layout.has_name("W"))
    assert not (layout.has_name("H"))
    assert not (layout.has_name("D"))
    assert layout.get_index_by_name("N") == 0
    assert layout.get_index_by_name("C") == 1

    layout = Layout("N...C")
    assert layout.has_name("N")
    assert not (layout.has_name("W"))
    assert not (layout.has_name("H"))
    assert not (layout.has_name("D"))
    assert layout.has_name("C")
    assert layout.get_index_by_name("C") == -1

    layout = Layout()
    assert not (layout.has_name("W"))
    assert not (layout.has_name("H"))
    assert not (layout.has_name("D"))
    assert not (layout.has_name("C"))

    layout = Layout("N...C")
    assert layout == "N...C"
    assert layout != "NC?"


def test_layout_helpers():
    layout = Layout("NCHWD")
    assert (layout_helpers.has_batch(layout))
    assert (layout_helpers.has_channels(layout))
    assert (layout_helpers.has_depth(layout))
    assert (layout_helpers.has_height(layout))
    assert (layout_helpers.has_width(layout))

    assert layout_helpers.batch_idx(layout) == 0
    assert layout_helpers.channels_idx(layout) == 1
    assert layout_helpers.height_idx(layout) == 2
    assert layout_helpers.width_idx(layout) == 3
    assert layout_helpers.depth_idx(layout) == 4

    layout = Layout("N...C")
    assert (layout_helpers.has_batch(layout))
    assert (layout_helpers.has_channels(layout))
    assert not (layout_helpers.has_depth(layout))
    assert not (layout_helpers.has_height(layout))
    assert not (layout_helpers.has_width(layout))

    assert layout_helpers.batch_idx(layout) == 0
    assert layout_helpers.channels_idx(layout) == -1

    with pytest.raises(RuntimeError):
        layout_helpers.height_idx(layout)

    with pytest.raises(RuntimeError):
        layout_helpers.width_idx(layout)

    with pytest.raises(RuntimeError):
        layout_helpers.depth_idx(layout)

    layout = Layout("NC?")
    assert (layout_helpers.has_batch(layout))
    assert (layout_helpers.has_channels(layout))
    assert not (layout_helpers.has_depth(layout))
    assert not (layout_helpers.has_height(layout))
    assert not (layout_helpers.has_width(layout))

    assert layout_helpers.batch_idx(layout) == 0
    assert layout_helpers.channels_idx(layout) == 1

    with pytest.raises(RuntimeError):
        layout_helpers.height_idx(layout)

    with pytest.raises(RuntimeError):
        layout_helpers.width_idx(layout)

    with pytest.raises(RuntimeError):
        layout_helpers.depth_idx(layout)
