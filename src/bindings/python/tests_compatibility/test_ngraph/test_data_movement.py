# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import ngraph as ng
from ngraph.impl import Type, Shape


def test_reverse_sequence():
    input_data = ng.parameter((2, 3, 4, 2), name="input_data", dtype=np.int32)
    seq_lengths = np.array([1, 2, 1, 2], dtype=np.int32)
    batch_axis = 2
    sequence_axis = 1

    input_param = ng.parameter(input_data.shape, name="input", dtype=np.int32)
    seq_lengths_param = ng.parameter(seq_lengths.shape, name="sequence lengths", dtype=np.int32)
    model = ng.reverse_sequence(input_param, seq_lengths_param, batch_axis, sequence_axis)

    assert model.get_type_name() == "ReverseSequence"
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == [2, 3, 4, 2]
    assert model.get_output_element_type(0) == Type.i32


def test_pad_edge():
    input_data = np.arange(1, 13).reshape([3, 4])
    pads_begin = np.array([0, 1], dtype=np.int32)
    pads_end = np.array([2, 3], dtype=np.int32)

    input_param = ng.parameter(input_data.shape, name="input", dtype=np.int32)
    model = ng.pad(input_param, pads_begin, pads_end, "edge")

    assert model.get_type_name() == "Pad"
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == [5, 8]
    assert model.get_output_element_type(0) == Type.i32


def test_pad_constant():
    input_data = np.arange(1, 13).reshape([3, 4])
    pads_begin = np.array([0, 1], dtype=np.int32)
    pads_end = np.array([2, 3], dtype=np.int32)

    input_param = ng.parameter(input_data.shape, name="input", dtype=np.int32)
    model = ng.pad(input_param, pads_begin, pads_end, "constant", arg_pad_value=np.array(100, dtype=np.int32))

    assert model.get_type_name() == "Pad"
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == [5, 8]
    assert model.get_output_element_type(0) == Type.i32


def test_select():
    cond = np.array([[False, False], [True, False], [True, True]])
    then_node = np.array([[-1, 0], [1, 2], [3, 4]], dtype=np.int32)
    else_node = np.array([[11, 10], [9, 8], [7, 6]], dtype=np.int32)

    node = ng.select(cond, then_node, else_node)
    assert node.get_type_name() == "Select"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [3, 2]
    assert node.get_output_element_type(0) == Type.i32


def test_gather_nd():
    indices_type = np.int32
    data_dtype = np.float32
    data = ng.parameter([2, 10, 80, 30, 50], dtype=data_dtype, name="data")
    indices = ng.parameter([2, 10, 30, 40, 2], dtype=indices_type, name="indices")
    batch_dims = 2
    expected_shape = [20, 30, 40, 50]

    node = ng.opset5.gather_nd(data, indices, batch_dims)
    assert node.get_type_name() == "GatherND"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape
    assert node.get_output_element_type(0) == Type.f32


def test_gather_v8_nd():
    indices_type = np.int32
    data_dtype = np.float32
    data = ng.parameter([2, 10, 80, 30, 50], dtype=data_dtype, name="data")
    indices = ng.parameter([2, 10, 30, 40, 2], dtype=indices_type, name="indices")
    batch_dims = 2
    expected_shape = [2, 10, 30, 40, 50]

    node = ng.gather_nd(data, indices, batch_dims)
    assert node.get_type_name() == "GatherND"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape
    assert node.get_output_element_type(0) == Type.f32


def test_gather_elements():
    indices_type = np.int32
    data_dtype = np.float32
    data = ng.parameter(Shape([2, 5]), dtype=data_dtype, name="data")
    indices = ng.parameter(Shape([2, 100]), dtype=indices_type, name="indices")
    axis = 1
    expected_shape = [2, 100]

    node = ng.gather_elements(data, indices, axis)
    assert node.get_type_name() == "GatherElements"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape
    assert node.get_output_element_type(0) == Type.f32
