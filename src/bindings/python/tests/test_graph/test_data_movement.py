# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import openvino.runtime.opset8 as ov
from openvino import Type, Shape


def test_reverse_sequence():
    input_data = ov.parameter((2, 3, 4, 2), name="input_data", dtype=np.int32)
    seq_lengths = np.array([1, 2, 1, 2], dtype=np.int32)
    batch_axis = 2
    sequence_axis = 1

    input_param = ov.parameter(input_data.shape, name="input", dtype=np.int32)
    seq_lengths_param = ov.parameter(seq_lengths.shape, name="sequence lengths", dtype=np.int32)
    model = ov.reverse_sequence(input_param, seq_lengths_param, batch_axis, sequence_axis)

    assert model.get_type_name() == "ReverseSequence"
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == [2, 3, 4, 2]
    assert model.get_output_element_type(0) == Type.i32


def test_select():
    cond = np.array([[False, False], [True, False], [True, True]])
    then_node = np.array([[-1, 0], [1, 2], [3, 4]], dtype=np.int32)
    else_node = np.array([[11, 10], [9, 8], [7, 6]], dtype=np.int32)

    node = ov.select(cond, then_node, else_node)
    assert node.get_type_name() == "Select"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [3, 2]
    assert node.get_output_element_type(0) == Type.i32


@pytest.mark.parametrize("op_name", ["Gather", "gatherv8", "gatherOpset8"])
def test_gather_v8_nd(op_name):
    data = ov.parameter([2, 10, 80, 30, 50], dtype=np.float32, name="data")
    indices = ov.parameter([2, 10, 30, 40, 2], dtype=np.int32, name="indices")
    batch_dims = 2

    node = ov.gather_nd(data, indices, batch_dims, name=op_name)
    assert node.get_type_name() == "GatherND"
    assert node.get_friendly_name() == op_name
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [2, 10, 30, 40, 50]
    assert node.get_output_element_type(0) == Type.f32


def test_gather_elements():
    indices_type = np.int32
    data_dtype = np.float32
    data = ov.parameter(Shape([2, 5]), dtype=data_dtype, name="data")
    indices = ov.parameter(Shape([2, 100]), dtype=indices_type, name="indices")
    axis = 1

    node = ov.gather_elements(data, indices, axis)
    assert node.get_type_name() == "GatherElements"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [2, 100]
    assert node.get_output_element_type(0) == Type.f32
