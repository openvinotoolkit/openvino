# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import ngraph as ng
from ngraph.impl import Type, Shape
from tests.runtime import get_runtime
from tests.test_ngraph.util import run_op_node


def test_reverse_sequence():
    input_data = np.array(
        [
            0,
            0,
            3,
            0,
            6,
            0,
            9,
            0,
            1,
            0,
            4,
            0,
            7,
            0,
            10,
            0,
            2,
            0,
            5,
            0,
            8,
            0,
            11,
            0,
            12,
            0,
            15,
            0,
            18,
            0,
            21,
            0,
            13,
            0,
            16,
            0,
            19,
            0,
            22,
            0,
            14,
            0,
            17,
            0,
            20,
            0,
            23,
            0,
        ],
        dtype=np.int32,
    ).reshape([2, 3, 4, 2])
    seq_lengths = np.array([1, 2, 1, 2], dtype=np.int32)
    batch_axis = 2
    sequence_axis = 1

    input_param = ng.parameter(input_data.shape, name="input", dtype=np.int32)
    seq_lengths_param = ng.parameter(seq_lengths.shape, name="sequence lengths", dtype=np.int32)
    model = ng.reverse_sequence(input_param, seq_lengths_param, batch_axis, sequence_axis)

    runtime = get_runtime()
    computation = runtime.computation(model, input_param, seq_lengths_param)
    result = computation(input_data, seq_lengths)

    expected = np.array(
        [
            0,
            0,
            4,
            0,
            6,
            0,
            10,
            0,
            1,
            0,
            3,
            0,
            7,
            0,
            9,
            0,
            2,
            0,
            5,
            0,
            8,
            0,
            11,
            0,
            12,
            0,
            16,
            0,
            18,
            0,
            22,
            0,
            13,
            0,
            15,
            0,
            19,
            0,
            21,
            0,
            14,
            0,
            17,
            0,
            20,
            0,
            23,
            0,
        ],
    ).reshape([1, 2, 3, 4, 2])
    assert np.allclose(result, expected)


def test_pad_edge():
    input_data = np.arange(1, 13).reshape([3, 4])
    pads_begin = np.array([0, 1], dtype=np.int32)
    pads_end = np.array([2, 3], dtype=np.int32)

    input_param = ng.parameter(input_data.shape, name="input", dtype=np.int32)
    model = ng.pad(input_param, pads_begin, pads_end, "edge")

    runtime = get_runtime()
    computation = runtime.computation(model, input_param)
    result = computation(input_data)

    expected = np.array(
        [
            [1, 1, 2, 3, 4, 4, 4, 4],
            [5, 5, 6, 7, 8, 8, 8, 8],
            [9, 9, 10, 11, 12, 12, 12, 12],
            [9, 9, 10, 11, 12, 12, 12, 12],
            [9, 9, 10, 11, 12, 12, 12, 12],
        ]
    )
    assert np.allclose(result, expected)


def test_pad_constant():
    input_data = np.arange(1, 13).reshape([3, 4])
    pads_begin = np.array([0, 1], dtype=np.int32)
    pads_end = np.array([2, 3], dtype=np.int32)

    input_param = ng.parameter(input_data.shape, name="input", dtype=np.int32)
    model = ng.pad(input_param, pads_begin, pads_end, "constant", arg_pad_value=np.array(100, dtype=np.int32))

    runtime = get_runtime()
    computation = runtime.computation(model, input_param)
    result = computation(input_data)

    expected = np.array(
        [
            [100, 1, 2, 3, 4, 100, 100, 100],
            [100, 5, 6, 7, 8, 100, 100, 100],
            [100, 9, 10, 11, 12, 100, 100, 100],
            [100, 100, 100, 100, 100, 100, 100, 100],
            [100, 100, 100, 100, 100, 100, 100, 100],
        ]
    )
    assert np.allclose(result, expected)


def test_select():
    cond = np.array([[False, False], [True, False], [True, True]])
    then_node = np.array([[-1, 0], [1, 2], [3, 4]], dtype=np.int32)
    else_node = np.array([[11, 10], [9, 8], [7, 6]], dtype=np.int32)
    excepted = np.array([[11, 10], [1, 8], [3, 4]], dtype=np.int32)

    result = run_op_node([cond, then_node, else_node], ng.select)
    assert np.allclose(result, excepted)


def test_gather_nd():
    indices_type = np.int32
    data_dtype = np.float32
    data = ng.parameter([2, 10, 80, 30, 50], dtype=data_dtype, name="data")
    indices = ng.parameter([2, 10, 30, 40, 2], dtype=indices_type, name="indices")
    batch_dims = 2
    expected_shape = [20, 30, 40, 50]

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
