# -*- coding: utf-8 -*-
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino.runtime.opset8 as ov
import numpy as np


def test_gather():
    input_data = ov.parameter((3, 3), name="input_data", dtype=np.float32)
    input_indices = ov.parameter((1, 2), name="input_indices", dtype=np.int32)
    input_axis = np.array([1], np.int32)
    expected_shape = [3, 1, 2]

    node = ov.gather(input_data, input_indices, input_axis)
    assert node.get_type_name() == "Gather"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape


def test_gather_with_scalar_axis():
    input_data = ov.parameter((3, 3), name="input_data", dtype=np.float32)
    input_indices = ov.parameter((1, 2), name="input_indices", dtype=np.int32)
    input_axis = np.array(1, np.int32)
    expected_shape = [3, 1, 2]

    node = ov.gather(input_data, input_indices, input_axis)
    assert node.get_type_name() == "Gather"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape


def test_gather_batch_dims_1():
    input_data = ov.parameter((2, 5), name="input_data", dtype=np.float32)
    input_indices = ov.parameter((2, 3), name="input_indices", dtype=np.int32)
    input_axis = np.array([1], np.int32)
    batch_dims = 1
    expected_shape = [2, 3]

    node = ov.gather(input_data, input_indices, input_axis, batch_dims)
    assert node.get_type_name() == "Gather"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape


def test_gather_negative_indices():
    input_data = ov.parameter((3, 3), name="input_data", dtype=np.float32)
    input_indices = ov.parameter((1, 2), name="input_indices", dtype=np.int32)
    input_axis = np.array([1], np.int32)
    expected_shape = [3, 1, 2]

    node = ov.gather(input_data, input_indices, input_axis)
    assert node.get_type_name() == "Gather"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape


def test_gather_batch_dims_1_negative_indices():
    input_data = ov.parameter((2, 5), name="input_data", dtype=np.float32)
    input_indices = ov.parameter((2, 3), name="input_indices", dtype=np.int32)
    input_axis = np.array([1], np.int32)
    batch_dims = 1
    expected_shape = [2, 3]

    node = ov.gather(input_data, input_indices, input_axis, batch_dims)
    assert node.get_type_name() == "Gather"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape
