# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino import Tensor, Type
import openvino.runtime.opset8 as ov
import numpy as np
import pytest


@pytest.mark.parametrize(("input_shape", "indices", "axis", "expected_shape", "batch_dims"), [
    ((3, 3), (1, 2), [1], [3, 1, 2], []),
    ((3, 3), (1, 2), 1, [3, 1, 2], []),
    ((2, 5), (2, 3), [1], [2, 3], [1]),
    ((2, 5), (2, 3), [1], [2, 2, 3], []),
])
def test_gather(input_shape, indices, axis, expected_shape, batch_dims):
    input_data = ov.parameter(input_shape, name="input_data", dtype=np.float32)
    input_indices = ov.parameter(indices, name="input_indices", dtype=np.int32)
    input_axis = np.array(axis, np.int32)

    node = ov.gather(input_data, input_indices, input_axis, *batch_dims)
    assert node.get_type_name() == "Gather"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape
    assert node.get_output_element_type(0) == Type.f32


@pytest.mark.parametrize(("data_str", "input_shape", "indices", "axis", "expected_shape", "batch_dims"), [
    (["Abc", " C de, Fghi.."], [2], [0], [0], [1], 0),
    (["Abc", " C de, Fghi.."], [1, 2], [1], [1], [1], 1),
])
@pytest.mark.parametrize("op_name", ["gather", "gatherOpset8"])
def test_gather_string(data_str, input_shape, indices, axis, expected_shape, batch_dims, op_name):
    input_data = np.array(data_str).reshape(input_shape)
    input_param = ov.parameter(input_shape, name="input_data", dtype=Type.string)

    input_indices = np.array(indices, np.int32)
    input_axis = np.array(axis, np.int32)

    node = ov.gather(input_param, input_indices, input_axis, batch_dims, name=op_name)
    out_tensor = Tensor(Type.string, input_shape)

    assert node.get_type_name() == "Gather"
    assert node.get_friendly_name() == op_name
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape
    assert node.get_output_element_type(0) == Type.string

    node.evaluate([out_tensor], [Tensor(input_data, shared_memory=False), Tensor(input_indices), Tensor(input_axis)])
    assert np.array(data_str[indices[0]]) == out_tensor.str_data
