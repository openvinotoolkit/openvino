# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import ngraph as ng
from ngraph.impl import Type


@pytest.mark.parametrize(
    ("ng_api_helper", "reduction_axes", "expected_shape"),
    [
        (ng.reduce_max, np.array([0, 1, 2, 3]), []),
        (ng.reduce_min, np.array([0, 1, 2, 3]), []),
        (ng.reduce_sum, np.array([0, 1, 2, 3]), []),
        (ng.reduce_prod, np.array([0, 1, 2, 3]), []),
        (ng.reduce_max, np.array([0]), [4, 3, 2]),
        (ng.reduce_min, np.array([0]), [4, 3, 2]),
        (ng.reduce_sum, np.array([0]), [4, 3, 2]),
        (ng.reduce_prod, np.array([0]), [4, 3, 2]),
        (ng.reduce_max, np.array([0, 2]), [4, 2]),
        (ng.reduce_min, np.array([0, 2]), [4, 2]),
        (ng.reduce_sum, np.array([0, 2]), [4, 2]),
        (ng.reduce_prod, np.array([0, 2]), [4, 2]),
    ],
)
def test_reduction_ops(ng_api_helper, reduction_axes, expected_shape):
    shape = [2, 4, 3, 2]
    np.random.seed(133391)
    input_data = np.random.randn(*shape).astype(np.float32)

    node = ng_api_helper(input_data, reduction_axes)
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape
    assert node.get_output_element_type(0) == Type.f32


@pytest.mark.parametrize(
    ("ng_api_helper", "reduction_axes", "expected_shape"),
    [
        (ng.reduce_logical_and, np.array([0]), [4, 3, 2]),
        (ng.reduce_logical_or, np.array([0]), [4, 3, 2]),
        (ng.reduce_logical_and, np.array([0, 2]), [4, 2]),
        (ng.reduce_logical_or, np.array([0, 2]), [4, 2]),
        (ng.reduce_logical_and, np.array([0, 1, 2, 3]), []),
        (ng.reduce_logical_or, np.array([0, 1, 2, 3]), []),
    ],
)
def test_reduction_logical_ops(ng_api_helper, reduction_axes, expected_shape):
    shape = [2, 4, 3, 2]
    np.random.seed(133391)
    input_data = np.random.randn(*shape).astype(bool)

    node = ng_api_helper(input_data, reduction_axes)
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape
    assert node.get_output_element_type(0) == Type.boolean


def test_topk():
    data_shape = [6, 12, 10, 24]
    data_parameter = ng.parameter(data_shape, name="Data", dtype=np.float32)
    K = np.int32(3)
    axis = np.int32(1)
    node = ng.topk(data_parameter, K, axis, "max", "value")
    assert node.get_type_name() == "TopK"
    assert node.get_output_size() == 2
    assert list(node.get_output_shape(0)) == [6, 3, 10, 24]
    assert list(node.get_output_shape(1)) == [6, 3, 10, 24]
    assert node.get_output_element_type(0) == Type.f32
    assert node.get_output_element_type(1) == Type.i32


@pytest.mark.parametrize(
    ("ng_api_helper", "reduction_axes", "expected_shape"),
    [
        (ng.reduce_mean, np.array([0, 1, 2, 3]), []),
        (ng.reduce_mean, np.array([0]), [4, 3, 2]),
        (ng.reduce_mean, np.array([0, 2]), [4, 2]),
    ],
)
def test_reduce_mean_op(ng_api_helper, reduction_axes, expected_shape):
    shape = [2, 4, 3, 2]
    np.random.seed(133391)
    input_data = np.random.randn(*shape).astype(np.float32)

    node = ng_api_helper(input_data, reduction_axes)
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape
    assert node.get_output_element_type(0) == Type.f32


def test_non_zero():

    data_shape = [3, 10, 100, 200]

    data_parameter = ng.parameter(data_shape, name="Data", dtype=np.float32)

    node = ng.non_zero(data_parameter)

    assert node.get_type_name() == "NonZero"
    assert node.get_output_size() == 1
    assert node.get_output_element_type(0) == Type.i64


def test_roi_align():

    data_shape = [7, 256, 200, 200]
    rois = [1000, 4]
    batch_indices = [1000]
    expected_shape = [1000, 256, 6, 6]

    data_parameter = ng.parameter(data_shape, name="Data", dtype=np.float32)
    rois_parameter = ng.parameter(rois, name="Rois", dtype=np.float32)
    batch_indices_parameter = ng.parameter(batch_indices, name="Batch_indices", dtype=np.int32)
    pooled_h = 6
    pooled_w = 6
    sampling_ratio = 2
    spatial_scale = np.float32(16)
    mode = "avg"

    node = ng.roi_align(
        data_parameter,
        rois_parameter,
        batch_indices_parameter,
        pooled_h,
        pooled_w,
        sampling_ratio,
        spatial_scale,
        mode,
    )

    assert node.get_type_name() == "ROIAlign"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape
    assert node.get_output_element_type(0) == Type.f32


@pytest.mark.parametrize(
    "input_shape, cumsum_axis, reverse",
    [([5, 2], 0, False), ([5, 2], 1, False), ([5, 2, 6], 2, False), ([5, 2], 0, True)],
)
def test_cum_sum(input_shape, cumsum_axis, reverse):
    input_data = np.arange(np.prod(input_shape), dtype=np.int64).reshape(input_shape)

    node = ng.cum_sum(input_data, cumsum_axis, reverse=reverse)
    assert node.get_output_size() == 1
    assert node.get_type_name() == "CumSum"
    assert list(node.get_output_shape(0)) == input_shape
    assert node.get_output_element_type(0) == Type.i64


def test_normalize_l2():
    input_shape = [1, 2, 3, 4]
    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)
    input_data += 1
    axes = np.array([1, 2, 3]).astype(np.int64)
    eps = 1e-6
    eps_mode = "add"

    node = ng.normalize_l2(input_data, axes, eps, eps_mode)
    assert node.get_output_size() == 1
    assert node.get_type_name() == "NormalizeL2"
    assert list(node.get_output_shape(0)) == input_shape
    assert node.get_output_element_type(0) == Type.f32
