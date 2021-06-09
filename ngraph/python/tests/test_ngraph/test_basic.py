# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import pytest

from _pyngraph import VariantInt, VariantString

import ngraph as ng
from ngraph.exceptions import UserInputError
from ngraph.impl import Function, PartialShape, Shape, Type
from ngraph.impl.op import Parameter
from tests.runtime import get_runtime
from tests.test_ngraph.util import run_op_node


def test_ngraph_function_api():
    shape = [2, 2]
    parameter_a = ng.parameter(shape, dtype=np.float32, name="A")
    parameter_b = ng.parameter(shape, dtype=np.float32, name="B")
    parameter_c = ng.parameter(shape, dtype=np.float32, name="C")
    model = (parameter_a + parameter_b) * parameter_c
    function = Function(model, [parameter_a, parameter_b, parameter_c], "TestFunction")

    function.get_parameters()[1].set_partial_shape(PartialShape([3, 4, 5]))

    ordered_ops = function.get_ordered_ops()
    op_types = [op.get_type_name() for op in ordered_ops]
    assert op_types == ["Parameter", "Parameter", "Parameter", "Add", "Multiply", "Result"]
    assert len(function.get_ops()) == 6
    assert function.get_output_size() == 1
    assert function.get_output_op(0).get_type_name() == "Result"
    assert function.get_output_element_type(0) == parameter_a.get_element_type()
    assert list(function.get_output_shape(0)) == [2, 2]
    assert (function.get_parameters()[1].get_partial_shape()) == PartialShape([3, 4, 5])
    assert len(function.get_parameters()) == 3
    assert len(function.get_results()) == 1
    assert function.get_friendly_name() == "TestFunction"


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
    ],
)
def test_simple_computation_on_ndarrays(dtype):
    runtime = get_runtime()

    shape = [2, 2]
    parameter_a = ng.parameter(shape, dtype=dtype, name="A")
    parameter_b = ng.parameter(shape, dtype=dtype, name="B")
    parameter_c = ng.parameter(shape, dtype=dtype, name="C")
    model = (parameter_a + parameter_b) * parameter_c
    computation = runtime.computation(model, parameter_a, parameter_b, parameter_c)

    value_a = np.array([[1, 2], [3, 4]], dtype=dtype)
    value_b = np.array([[5, 6], [7, 8]], dtype=dtype)
    value_c = np.array([[2, 3], [4, 5]], dtype=dtype)
    result = computation(value_a, value_b, value_c)
    assert np.allclose(result, np.array([[12, 24], [40, 60]], dtype=dtype))

    value_a = np.array([[9, 10], [11, 12]], dtype=dtype)
    value_b = np.array([[13, 14], [15, 16]], dtype=dtype)
    value_c = np.array([[5, 4], [3, 2]], dtype=dtype)
    result = computation(value_a, value_b, value_c)
    assert np.allclose(result, np.array([[110, 96], [78, 56]], dtype=dtype))


def test_serialization():
    dtype = np.float32
    shape = [2, 2]
    parameter_a = ng.parameter(shape, dtype=dtype, name="A")
    parameter_b = ng.parameter(shape, dtype=dtype, name="B")
    parameter_c = ng.parameter(shape, dtype=dtype, name="C")
    model = (parameter_a + parameter_b) * parameter_c

    runtime = get_runtime()
    computation = runtime.computation(model, parameter_a, parameter_b, parameter_c)
    try:
        serialized = computation.serialize(2)
        serial_json = json.loads(serialized)

        assert serial_json[0]["name"] != ""
        assert 10 == len(serial_json[0]["ops"])
    except Exception:
        pass


def test_broadcast_1():
    input_data = np.array([1, 2, 3], dtype=np.int32)

    new_shape = [3, 3]
    expected = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
    result = run_op_node([input_data], ng.broadcast, new_shape)
    assert np.allclose(result, expected)


def test_broadcast_2():
    input_data = np.arange(4, dtype=np.int32)
    new_shape = [3, 4, 2, 4]
    expected = np.broadcast_to(input_data, new_shape)
    result = run_op_node([input_data], ng.broadcast, new_shape)
    assert np.allclose(result, expected)


def test_broadcast_3():
    input_data = np.array([1, 2, 3], dtype=np.int32)
    new_shape = [3, 3]
    axis_mapping = [0]
    expected = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]

    result = run_op_node([input_data], ng.broadcast, new_shape, axis_mapping, "EXPLICIT")
    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    "destination_type, input_data",
    [(bool, np.zeros((2, 2), dtype=np.int32)), ("boolean", np.zeros((2, 2), dtype=np.int32))],
)
def test_convert_to_bool(destination_type, input_data):
    expected = np.array(input_data, dtype=bool)
    result = run_op_node([input_data], ng.convert, destination_type)
    assert np.allclose(result, expected)
    assert np.array(result).dtype == bool


@pytest.mark.parametrize(
    "destination_type, rand_range, in_dtype, expected_type",
    [
        pytest.param(np.float32, (-8, 8), np.int32, np.float32),
        pytest.param(np.float64, (-16383, 16383), np.int64, np.float64),
        pytest.param("f32", (-8, 8), np.int32, np.float32),
        pytest.param("f64", (-16383, 16383), np.int64, np.float64),
    ],
)
def test_convert_to_float(destination_type, rand_range, in_dtype, expected_type):
    np.random.seed(133391)
    input_data = np.random.randint(*rand_range, size=(2, 2), dtype=in_dtype)
    expected = np.array(input_data, dtype=expected_type)
    result = run_op_node([input_data], ng.convert, destination_type)
    assert np.allclose(result, expected)
    assert np.array(result).dtype == expected_type


@pytest.mark.parametrize(
    "destination_type, expected_type",
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
    input_data = (np.ceil(-8 + np.random.rand(2, 3, 4) * 16)).astype(np.float32)
    expected = np.array(input_data, dtype=expected_type)
    result = run_op_node([input_data], ng.convert, destination_type)
    assert np.allclose(result, expected)
    assert np.array(result).dtype == expected_type


@pytest.mark.parametrize(
    "destination_type, expected_type",
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
    input_data = np.ceil(np.random.rand(2, 3, 4) * 16).astype(np.float32)
    expected = np.array(input_data, dtype=expected_type)
    result = run_op_node([input_data], ng.convert, destination_type)
    assert np.allclose(result, expected)
    assert np.array(result).dtype == expected_type


def test_bad_data_shape():
    A = ng.parameter(shape=[2, 2], name="A", dtype=np.float32)
    B = ng.parameter(shape=[2, 2], name="B")
    model = A + B
    runtime = get_runtime()
    computation = runtime.computation(model, A, B)

    value_a = np.array([[1, 2]], dtype=np.float32)
    value_b = np.array([[5, 6], [7, 8]], dtype=np.float32)
    with pytest.raises(UserInputError):
        computation(value_a, value_b)


def test_constant_get_data_bool():
    input_data = np.array([True, False, False, True])
    node = ng.constant(input_data, dtype=np.bool)
    retrieved_data = node.get_data()
    assert np.allclose(input_data, retrieved_data)


@pytest.mark.parametrize("data_type", [np.float32, np.float64])
def test_constant_get_data_floating_point(data_type):
    np.random.seed(133391)
    input_data = np.random.randn(2, 3, 4).astype(data_type)
    min_value = -1.0e20
    max_value = 1.0e20
    input_data = min_value + input_data * max_value * data_type(2)
    node = ng.constant(input_data, dtype=data_type)
    retrieved_data = node.get_data()
    assert np.allclose(input_data, retrieved_data)


@pytest.mark.parametrize("data_type", [np.int64, np.int32, np.int16, np.int8])
def test_constant_get_data_signed_integer(data_type):
    np.random.seed(133391)
    input_data = np.random.randint(
        np.iinfo(data_type).min, np.iinfo(data_type).max, size=[2, 3, 4], dtype=data_type
    )
    node = ng.constant(input_data, dtype=data_type)
    retrieved_data = node.get_data()
    assert np.allclose(input_data, retrieved_data)


@pytest.mark.parametrize("data_type", [np.uint64, np.uint32, np.uint16, np.uint8])
def test_constant_get_data_unsigned_integer(data_type):
    np.random.seed(133391)
    input_data = np.random.randn(2, 3, 4).astype(data_type)
    input_data = (
        np.iinfo(data_type).min + input_data * np.iinfo(data_type).max + input_data * np.iinfo(data_type).max
    )
    node = ng.constant(input_data, dtype=data_type)
    retrieved_data = node.get_data()
    assert np.allclose(input_data, retrieved_data)


def test_result():
    node = np.array([[11, 10], [1, 8], [3, 4]])
    result = run_op_node([node], ng.result)
    assert np.allclose(result, node)


def test_node_friendly_name():
    dummy_node = ng.parameter(shape=[1], name="dummy_name")

    assert(dummy_node.friendly_name == "dummy_name")

    dummy_node.set_friendly_name("changed_name")

    assert(dummy_node.get_friendly_name() == "changed_name")

    dummy_node.friendly_name = "new_name"

    assert(dummy_node.get_friendly_name() == "new_name")


def test_node_output():
    input_array = np.array([0, 1, 2, 3, 4, 5])
    splits = 3
    expected_shape = len(input_array) // splits

    input_tensor = ng.constant(input_array, dtype=np.int32)
    axis = ng.constant(0, dtype=np.int64)
    split_node = ng.split(input_tensor, axis, splits)

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


def test_node_input():
    shape = [2, 2]
    parameter_a = ng.parameter(shape, dtype=np.float32, name="A")
    parameter_b = ng.parameter(shape, dtype=np.float32, name="B")

    model = parameter_a + parameter_b

    model_inputs = model.inputs()

    assert len(model_inputs) == 2
    assert [input_node.get_index() for input_node in model_inputs] == [0, 1]
    assert np.equal(
        [input_node.get_element_type() for input_node in model_inputs],
        model.get_element_type(),
    ).all()
    assert np.equal(
        [input_node.get_shape() for input_node in model_inputs], Shape(shape)
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
    parameter_a = ng.parameter(shape, dtype=np.float32, name="A")
    parameter_b = ng.parameter(shape, dtype=np.float32, name="B")

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


def test_variants():
    variant_int = VariantInt(32)
    variant_str = VariantString("test_text")

    assert variant_int.get() == 32
    assert variant_str.get() == "test_text"

    variant_int.set(777)
    variant_str.set("another_text")

    assert variant_int.get() == 777
    assert variant_str.get() == "another_text"


def test_runtime_info():
    test_shape = PartialShape([1, 1, 1, 1])
    test_type = Type.f32
    test_param = Parameter(test_type, test_shape)
    relu_node = ng.relu(test_param)
    runtime_info = relu_node.get_rt_info()
    runtime_info["affinity"] = "test_affinity"
    relu_node.set_friendly_name("testReLU")
    runtime_info_after = relu_node.get_rt_info()

    assert runtime_info_after["affinity"] == "test_affinity"


def test_mutiple_outputs():
    input_shape = [4, 4]
    input_data = np.arange(-8, 8).reshape(input_shape)

    expected_output = np.split(input_data, 2, axis=1)[0]
    expected_output[expected_output < 0] = 0

    test_param = ng.parameter(input_shape, dtype=np.float32, name="A")
    split = ng.split(test_param, axis=1, num_splits=2)
    split_first_output = split.output(0)
    relu = ng.relu(split_first_output)

    runtime = get_runtime()
    computation = runtime.computation(relu, test_param)
    output = computation(input_data)

    assert np.equal(output, expected_output).all()


def test_sink_function_ctor():
    input_data = ng.parameter([2, 2], name="input_data", dtype=np.float32)
    rv = ng.read_value(input_data, "var_id_667")
    add = ng.add(rv, input_data, name="MemoryAdd")
    node = ng.assign(add, "var_id_667")
    res = ng.result(add, "res")
    function = Function(results=[res], sinks=[node], parameters=[input_data], name="TestFunction")

    ordered_ops = function.get_ordered_ops()
    op_types = [op.get_type_name() for op in ordered_ops]
    assert op_types == ["Parameter", "ReadValue", "Add", "Assign", "Result"]
    assert len(function.get_ops()) == 5
    assert function.get_output_size() == 1
    assert function.get_output_op(0).get_type_name() == "Result"
    assert function.get_output_element_type(0) == input_data.get_element_type()
    assert list(function.get_output_shape(0)) == [2, 2]
    assert (function.get_parameters()[0].get_partial_shape()) == PartialShape([2, 2])
    assert len(function.get_parameters()) == 1
    assert len(function.get_results()) == 1
    assert function.get_friendly_name() == "TestFunction"
