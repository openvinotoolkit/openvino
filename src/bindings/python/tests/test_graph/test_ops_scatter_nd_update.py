# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from openvino import PartialShape, Type
from openvino.runtime import opset4, opset15

scatter_version_opset = pytest.mark.parametrize("opset", [opset4, opset15])


@scatter_version_opset
def test_scatter_nd_update(opset):
    data_shape = [4, 4, 4]
    indices_shape = [2, 1]
    updates_shape = [2, 4, 4]

    data_param = opset.parameter(shape=data_shape, dtype=Type.f32, name="data")
    indices_param = opset.parameter(shape=indices_shape, dtype=Type.i32, name="indices")
    updates_param = opset.parameter(shape=updates_shape, dtype=Type.f32, name="updates")

    scatter_nd_node = opset.scatter_nd_update(data_param, indices_param, updates_param)

    assert scatter_nd_node.get_type_name() == "ScatterNDUpdate"
    assert scatter_nd_node.get_output_size() == 1
    assert scatter_nd_node.get_output_partial_shape(0).same_scheme(PartialShape(data_shape))
    assert scatter_nd_node.get_output_element_type(0) == Type.f32


@scatter_version_opset
def test_scatter_nd_update_basic(opset):
    data = np.array([1, 2, 3, 4, 5])
    indices = np.array([[0], [2]])
    updates = np.array([9, 10])

    scatter_nd_node = opset.scatter_nd_update(data, indices, updates)
    assert scatter_nd_node.get_type_name() == "ScatterNDUpdate"
    assert scatter_nd_node.get_output_size() == 1
    assert scatter_nd_node.get_output_partial_shape(0).same_scheme(PartialShape(data.shape))
    assert scatter_nd_node.get_output_element_type(0) == Type(data.dtype)


@scatter_version_opset
def test_scatter_nd_update_multidimensional(opset):
    data = np.array([[1, 2], [3, 4]])
    indices = np.array([[0, 1], [1, 0]])
    updates = np.array([9, 10])

    scatter_nd_node = opset.scatter_nd_update(data, indices, updates)
    assert scatter_nd_node.get_type_name() == "ScatterNDUpdate"
    assert scatter_nd_node.get_output_size() == 1
    assert scatter_nd_node.get_output_partial_shape(0).same_scheme(PartialShape(data.shape))
    assert scatter_nd_node.get_output_element_type(0) == Type(data.dtype)


@scatter_version_opset
def test_scatter_nd_update_mismatched_updates_shape(opset):
    data = np.array([1, 2, 3])
    indices = np.array([[0], [1]])
    updates = np.array([4])

    with pytest.raises(RuntimeError):
        opset.scatter_nd_update(data, indices, updates)


@scatter_version_opset
def test_scatter_nd_update_non_integer_indices(opset):
    data = np.array([1, 2, 3])
    indices = np.array([[0.5]])
    updates = np.array([4])

    with pytest.raises(RuntimeError):
        opset.scatter_nd_update(data, indices, updates)


@scatter_version_opset
def test_scatter_nd_update_negative_indices(opset):
    data = np.array([1, 2, 3, 4])
    indices = np.array([[-1]])
    updates = np.array([5])

    scatter_nd_node = opset.scatter_nd_update(data, indices, updates)
    assert scatter_nd_node.get_type_name() == "ScatterNDUpdate"
    assert scatter_nd_node.get_output_size() == 1
    assert scatter_nd_node.get_output_partial_shape(0).same_scheme(PartialShape(data.shape))
    assert scatter_nd_node.get_output_element_type(0) == Type(data.dtype)


@scatter_version_opset
def test_scatter_nd_update_multi_index_per_update(opset):
    data = np.array([[1, 2], [3, 4]])
    indices = np.array([[0, 0], [0, 1]])
    updates = np.array([5, 6])

    scatter_nd_node = opset.scatter_nd_update(data, indices, updates)
    assert scatter_nd_node.get_type_name() == "ScatterNDUpdate"
    assert scatter_nd_node.get_output_size() == 1
    assert scatter_nd_node.get_output_partial_shape(0).same_scheme(PartialShape(data.shape))
    assert scatter_nd_node.get_output_element_type(0) == Type(data.dtype)


@scatter_version_opset
def test_scatter_nd_update_non_contiguous_indices(opset):
    data = np.array([10, 20, 30, 40, 50])
    indices = np.array([[0], [3]])
    updates = np.array([100, 400])

    scatter_nd_node = opset.scatter_nd_update(data, indices, updates)
    assert scatter_nd_node.get_type_name() == "ScatterNDUpdate"
    assert scatter_nd_node.get_output_size() == 1
    assert scatter_nd_node.get_output_partial_shape(0).same_scheme(PartialShape(data.shape))
    assert scatter_nd_node.get_output_element_type(0) == Type(data.dtype)


@scatter_version_opset
def test_scatter_nd_update_large_updates(opset):
    data = np.zeros(1000, dtype=np.float64)
    indices = np.reshape(np.arange(1000), (-1, 1))
    updates = np.arange(1000, dtype=np.float64)

    scatter_nd_node = opset.scatter_nd_update(data, indices, updates)
    assert scatter_nd_node.get_type_name() == "ScatterNDUpdate"
    assert scatter_nd_node.get_output_size() == 1
    assert scatter_nd_node.get_output_partial_shape(0).same_scheme(PartialShape(data.shape))
    assert scatter_nd_node.get_output_element_type(0) == Type(data.dtype)


@scatter_version_opset
def test_scatter_nd_update_opseterlapping_indices(opset):
    data = np.array([1, 2, 3, 4, 5])
    indices = np.array([[1], [1], [3]])
    updates = np.array([10, 20, 30])

    scatter_nd_node = opset.scatter_nd_update(data, indices, updates)
    assert scatter_nd_node.get_type_name() == "ScatterNDUpdate"
    assert scatter_nd_node.get_output_size() == 1
    assert scatter_nd_node.get_output_partial_shape(0).same_scheme(PartialShape(data.shape))
    assert scatter_nd_node.get_output_element_type(0) == Type(data.dtype)


@scatter_version_opset
def test_scatter_nd_update_3d_data(opset):
    data = np.zeros((2, 2, 2), dtype=np.float64)
    indices = np.array([[0, 0, 1], [1, 1, 0]])
    updates = np.array([1, 2], dtype=np.float64)

    scatter_nd_node = opset.scatter_nd_update(data, indices, updates)
    assert scatter_nd_node.get_type_name() == "ScatterNDUpdate"
    assert scatter_nd_node.get_output_size() == 1
    assert scatter_nd_node.get_output_partial_shape(0).same_scheme(PartialShape(data.shape))
    assert scatter_nd_node.get_output_element_type(0) == Type(data.dtype)


@scatter_version_opset
def test_scatter_nd_update_all_indices(opset):
    data = np.ones((2, 3), dtype=np.float64)
    indices = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]])
    updates = np.array([10, 20, 30, 40, 50, 60], dtype=np.float64)

    scatter_nd_node = opset.scatter_nd_update(data, indices, updates)
    assert scatter_nd_node.get_type_name() == "ScatterNDUpdate"
    assert scatter_nd_node.get_output_size() == 1
    assert scatter_nd_node.get_output_partial_shape(0).same_scheme(PartialShape(data.shape))
    assert scatter_nd_node.get_output_element_type(0) == Type(data.dtype)


@scatter_version_opset
def test_scatter_nd_update_invalid_updates_shape(opset):
    data = np.array([1, 2, 3, 4])
    indices = np.array([[1], [2]])
    updates = np.array([5])

    with pytest.raises(RuntimeError):
        opset.scatter_nd_update(data, indices, updates)


@scatter_version_opset
def test_scatter_nd_update_negative_updates(opset):
    data = np.array([1, 2, 3, 4, 5])
    indices = np.array([[1], [3]])
    updates = np.array([-1, -2])

    scatter_nd_node = opset.scatter_nd_update(data, indices, updates)
    assert scatter_nd_node.get_type_name() == "ScatterNDUpdate"
    assert scatter_nd_node.get_output_size() == 1
    assert scatter_nd_node.get_output_partial_shape(0).same_scheme(PartialShape(data.shape))
    assert scatter_nd_node.get_output_element_type(0) == Type(data.dtype)


@scatter_version_opset
def test_scatter_nd_update_empty_indices_and_updates(opset):
    data = np.array([1, 2, 3], dtype=np.float64)
    indices = np.array([], dtype=np.int64).reshape(0, 1)
    updates = np.array([], dtype=np.float64)

    scatter_nd_node = opset.scatter_nd_update(data, indices, updates)
    assert scatter_nd_node.get_type_name() == "ScatterNDUpdate"
    assert scatter_nd_node.get_output_size() == 1
    assert scatter_nd_node.get_output_partial_shape(0).same_scheme(PartialShape(data.shape))
    assert scatter_nd_node.get_output_element_type(0) == Type(data.dtype)


@pytest.mark.parametrize(
    "reduction",
    [
        None,
        "none",
        "sUm",
        "SUB",
        "pRod",
        "miN",
        "Max",
    ],
)
def test_scatter_nd_update_reduction(reduction):
    data = np.array([1, 2, 3, 4, 5])
    indices = np.array([[0], [2]])
    updates = np.array([9, 10])

    scatter_nd_node = opset15.scatter_nd_update(data, indices, updates, reduction)
    assert scatter_nd_node.get_type_name() == "ScatterNDUpdate"
    op_attrs = scatter_nd_node.get_attributes()
    if reduction is None:
        assert op_attrs["reduction"] == "none"
    else:
        assert op_attrs["reduction"] == reduction.lower()
    assert scatter_nd_node.get_output_size() == 1
    assert scatter_nd_node.get_output_partial_shape(0).same_scheme(PartialShape(data.shape))
    assert scatter_nd_node.get_output_element_type(0) == Type(data.dtype)
