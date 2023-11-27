# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from openvino.runtime import PartialShape

import openvino.runtime.opset13 as ov
from openvino.runtime import Type


def test_scatter_nd_update():
    data_shape = [4, 4, 4]
    indices_shape = [2, 1]
    updates_shape = [2, 4, 4]

    data_param = ov.parameter(shape=data_shape, dtype=Type.f32, name="data")
    indices_param = ov.parameter(shape=indices_shape, dtype=Type.i32, name="indices")
    updates_param = ov.parameter(shape=updates_shape, dtype=Type.f32, name="updates")

    scatter_nd_node = ov.scatter_nd_update(data_param, indices_param, updates_param)

    assert scatter_nd_node.get_type_name() == "ScatterNDUpdate"
    assert scatter_nd_node.get_output_size() == 1
    assert scatter_nd_node.get_output_partial_shape(0).same_scheme(PartialShape(data_shape))
    assert scatter_nd_node.get_output_element_type(0) == Type.f32


def test_scatter_nd_update_basic():
    data = np.array([1, 2, 3, 4, 5])
    indices = np.array([[0], [2]])
    updates = np.array([9, 10])

    result = ov.scatter_nd_update(data, indices, updates)
    expected = np.array([9, 2, 10, 4, 5])
    np.testing.assert_array_equal(result, expected)


def test_scatter_nd_update_multidimensional():
    data = np.array([[1, 2], [3, 4]])
    indices = np.array([[0, 1], [1, 0]])
    updates = np.array([9, 10])

    result = ov.scatter_nd_update(data, indices, updates)
    expected = np.array([[1, 9], [10, 4]])
    np.testing.assert_array_equal(result, expected)


def test_scatter_nd_update_mismatched_updates_shape():
    data = np.array([1, 2, 3])
    indices = np.array([[0], [1]])
    updates = np.array([4])

    with pytest.raises(RuntimeError):
        ov.scatter_nd_update(data, indices, updates)


def test_scatter_nd_update_non_integer_indices():
    data = np.array([1, 2, 3])
    indices = np.array([[0.5]])
    updates = np.array([4])

    with pytest.raises(RuntimeError):
        ov.scatter_nd_update(data, indices, updates)


def test_scatter_nd_update_negative_indices():
    data = np.array([1, 2, 3, 4])
    indices = np.array([[-1]])
    updates = np.array([5])

    result = ov.scatter_nd_update(data, indices, updates)
    expected = np.array([1, 2, 3, 5])
    np.testing.assert_array_equal(result, expected)


def test_scatter_nd_update_multi_index_per_update():
    data = np.array([[1, 2], [3, 4]])
    indices = np.array([[0, 0], [0, 1]])
    updates = np.array([5, 6])

    result = ov.scatter_nd_update(data, indices, updates)
    expected = np.array([[5, 6], [3, 4]])
    np.testing.assert_array_equal(result, expected)


def test_scatter_nd_update_non_contiguous_indices():
    data = np.array([10, 20, 30, 40, 50])
    indices = np.array([[0], [3]])
    updates = np.array([100, 400])

    result = ov.scatter_nd_update(data, indices, updates)
    expected = np.array([100, 20, 30, 400, 50])
    np.testing.assert_array_equal(result, expected)


def test_scatter_nd_update_large_updates():
    data = np.zeros(1000, dtype=np.float64)
    indices = np.reshape(np.arange(1000), (-1, 1))
    updates = np.arange(1000, dtype=np.float64)

    result = ov.scatter_nd_update(data, indices, updates)
    expected = np.arange(1000, dtype=np.float64)
    np.testing.assert_array_equal(result, expected)


def test_scatter_nd_update_overlapping_indices():
    data = np.array([1, 2, 3, 4, 5])
    indices = np.array([[1], [1], [3]])
    updates = np.array([10, 20, 30])

    result = ov.scatter_nd_update(data, indices, updates)
    expected = np.array([1, 20, 3, 30, 5])
    np.testing.assert_array_equal(result, expected)


def test_scatter_nd_update_3d_data():
    data = np.zeros((2, 2, 2), dtype=np.float64)
    indices = np.array([[0, 0, 1], [1, 1, 0]])
    updates = np.array([1, 2], dtype=np.float64)

    result = ov.scatter_nd_update(data, indices, updates)
    expected = np.array([[[0, 1], [0, 0]], [[0, 0], [2, 0]]], dtype=np.float64)
    np.testing.assert_array_equal(result, expected)


def test_scatter_nd_update_all_indices():
    data = np.ones((2, 3), dtype=np.float64)
    indices = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]])
    updates = np.array([10, 20, 30, 40, 50, 60], dtype=np.float64)

    result = ov.scatter_nd_update(data, indices, updates)
    expected = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.float64)
    np.testing.assert_array_equal(result, expected)


def test_scatter_nd_update_invalid_updates_shape():
    data = np.array([1, 2, 3, 4])
    indices = np.array([[1], [2]])
    updates = np.array([5])

    with pytest.raises(RuntimeError):
        ov.scatter_nd_update(data, indices, updates)


def test_scatter_nd_update_negative_updates():
    data = np.array([1, 2, 3, 4, 5])
    indices = np.array([[1], [3]])
    updates = np.array([-1, -2])

    result = ov.scatter_nd_update(data, indices, updates)
    expected = np.array([1, -1, 3, -2, 5])
    np.testing.assert_array_equal(result, expected)


def test_scatter_nd_update_empty_indices_and_updates():
    data = np.array([1, 2, 3], dtype=np.int64)
    indices = np.array([], dtype=np.int64).reshape(0, 1)
    updates = np.array([], dtype=np.int64)

    result = ov.scatter_nd_update(data, indices, updates)
    expected = np.array([1, 2, 3], dtype=np.int64)
    np.testing.assert_array_equal(result, expected)
