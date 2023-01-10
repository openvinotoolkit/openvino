# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import ngraph as ng
import numpy as np


def test_gather():
    input_data = ng.parameter((3, 3), name="input_data", dtype=np.float32)
    input_indices = ng.parameter((1, 2), name="input_indices", dtype=np.int32)
    input_axis = np.array([1], np.int32)

    node = ng.gather(input_data, input_indices, input_axis)
    assert node.get_type_name() == "Gather"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [3, 1, 2]


def test_gather_with_scalar_axis():
    input_data = ng.parameter((3, 3), name="input_data", dtype=np.float32)
    input_indices = ng.parameter((1, 2), name="input_indices", dtype=np.int32)
    input_axis = np.array(1, np.int32)

    node = ng.gather(input_data, input_indices, input_axis)
    assert node.get_type_name() == "Gather"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [3, 1, 2]


def test_gather_batch_dims_1():
    input_data = ng.parameter((2, 5), name="input_data", dtype=np.float32)
    input_indices = ng.parameter((2, 3), name="input_indices", dtype=np.int32)
    input_axis = np.array([1], np.int32)
    batch_dims = 1

    node = ng.gather(input_data, input_indices, input_axis, batch_dims)
    assert node.get_type_name() == "Gather"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [2, 3]


def test_gather_negative_indices():
    input_data = ng.parameter((3, 3), name="input_data", dtype=np.float32)
    input_indices = ng.parameter((1, 2), name="input_indices", dtype=np.int32)
    input_axis = np.array([1], np.int32)

    node = ng.gather(input_data, input_indices, input_axis)
    assert node.get_type_name() == "Gather"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [3, 1, 2]


def test_gather_batch_dims_1_negative_indices():
    input_data = ng.parameter((2, 5), name="input_data", dtype=np.float32)
    input_indices = ng.parameter((2, 3), name="input_indices", dtype=np.int32)
    input_axis = np.array([1], np.int32)
    batch_dims = 1

    node = ng.gather(input_data, input_indices, input_axis, batch_dims)
    assert node.get_type_name() == "Gather"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [2, 3]
