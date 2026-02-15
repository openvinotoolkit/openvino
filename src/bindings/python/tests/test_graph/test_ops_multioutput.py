# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import openvino.opset8 as ov


def test_split():
    input_tensor = ov.constant(np.array([0, 1, 2, 3, 4, 5], dtype=np.int32))
    axis = ov.constant(0, dtype=np.int64)
    splits = 3

    split_node = ov.split(input_tensor, axis, splits)
    assert split_node.get_type_name() == "Split"
    assert split_node.get_output_size() == 3
    assert list(split_node.get_output_shape(0)) == [2]
    assert list(split_node.get_output_shape(1)) == [2]
    assert list(split_node.get_output_shape(2)) == [2]


def test_variadic_split():
    input_tensor = ov.constant(np.array([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]], dtype=np.int32))
    axis = ov.constant(1, dtype=np.int64)
    splits = ov.constant(np.array([2, 4], dtype=np.int64))

    v_split_node = ov.variadic_split(input_tensor, axis, splits)
    assert v_split_node.get_type_name() == "VariadicSplit"
    assert v_split_node.get_output_size() == 2
    assert list(v_split_node.get_output_shape(0)) == [2, 2]
    assert list(v_split_node.get_output_shape(1)) == [2, 4]
