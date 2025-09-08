# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import openvino.opset10 as ov


@pytest.mark.parametrize(
    ("graph_api_helper", "reduction_axes", "expected_shape"),
    [
        (ov.reduce_max, np.array([0, 1, 2, 3]), []),
        (ov.reduce_min, np.array([0, 1, 2, 3]), []),
        (ov.reduce_sum, np.array([0, 1, 2, 3]), []),
        (ov.reduce_prod, np.array([0, 1, 2, 3]), []),
        (ov.reduce_max, np.array([0]), [4, 3, 2]),
        (ov.reduce_min, np.array([0]), [4, 3, 2]),
        (ov.reduce_sum, np.array([0]), [4, 3, 2]),
        (ov.reduce_prod, np.array([0]), [4, 3, 2]),
        (ov.reduce_max, np.array([0, 2]), [4, 2]),
        (ov.reduce_min, np.array([0, 2]), [4, 2]),
        (ov.reduce_sum, np.array([0, 2]), [4, 2]),
        (ov.reduce_prod, np.array([0, 2]), [4, 2]),
    ],
)
def test_reduction_ops(graph_api_helper, reduction_axes, expected_shape):
    shape = [2, 4, 3, 2]
    np.random.seed(133391)
    input_data = np.random.randn(*shape).astype(np.float32)

    node = graph_api_helper(input_data, reduction_axes)
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape


@pytest.mark.parametrize(
    ("graph_api_helper", "reduction_axes", "expected_shape"),
    [
        (ov.reduce_logical_and, np.array([0]), [4, 3, 2]),
        (ov.reduce_logical_or, np.array([0]), [4, 3, 2]),
        (ov.reduce_logical_and, np.array([0, 2]), [4, 2]),
        (ov.reduce_logical_or, np.array([0, 2]), [4, 2]),
        (ov.reduce_logical_and, np.array([0, 1, 2, 3]), []),
        (ov.reduce_logical_or, np.array([0, 1, 2, 3]), []),
    ],
)
def test_reduction_logical_ops(graph_api_helper, reduction_axes, expected_shape):
    shape = [2, 4, 3, 2]
    np.random.seed(133391)
    input_data = np.random.randn(*shape).astype(bool)

    node = graph_api_helper(input_data, reduction_axes)
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape


def test_topk():
    data_shape = [6, 12, 10, 24]
    data_parameter = ov.parameter(data_shape, name="Data", dtype=np.float32)
    k_val = np.int32(3)
    axis = np.int32(1)
    node = ov.topk(data_parameter, k_val, axis, "max", "value")
    assert node.get_type_name() == "TopK"
    assert node.get_output_size() == 2
    assert list(node.get_output_shape(0)) == [6, 3, 10, 24]
    assert list(node.get_output_shape(1)) == [6, 3, 10, 24]


@pytest.mark.parametrize(
    ("graph_api_helper", "reduction_axes", "expected_shape"),
    [
        (ov.reduce_mean, np.array([0, 1, 2, 3]), []),
        (ov.reduce_mean, np.array([0]), [4, 3, 2]),
        (ov.reduce_mean, np.array([0, 2]), [4, 2]),
    ],
)
def test_reduce_mean_op(graph_api_helper, reduction_axes, expected_shape):
    shape = [2, 4, 3, 2]
    np.random.seed(133391)
    input_data = np.random.randn(*shape).astype(np.float32)

    node = graph_api_helper(input_data, reduction_axes)
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape


def test_non_zero():

    data_shape = [3, 10, 100, 200]

    data_parameter = ov.parameter(data_shape, name="Data", dtype=np.float32)

    node = ov.non_zero(data_parameter)

    assert node.get_type_name() == "NonZero"
    assert node.get_output_size() == 1


def test_roi_align():

    data_shape = [7, 256, 200, 200]
    rois = [1000, 4]
    batch_indices = [1000]

    data_parameter = ov.parameter(data_shape, name="Data", dtype=np.float32)
    rois_parameter = ov.parameter(rois, name="Rois", dtype=np.float32)
    batch_indices_parameter = ov.parameter(batch_indices, name="Batch_indices", dtype=np.int32)
    pooled_h = 6
    pooled_w = 6
    sampling_ratio = 2
    spatial_scale = np.float32(16)
    mode = "avg"

    node = ov.roi_align(
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
    assert list(node.get_output_shape(0)) == [1000, 256, 6, 6]


@pytest.mark.parametrize(
    ("input_shape", "cumsum_axis", "reverse"),
    [([5, 2], 0, False), ([5, 2], 1, False), ([5, 2, 6], 2, False), ([5, 2], 0, True)],
)
def test_cum_sum(input_shape, cumsum_axis, reverse):
    input_data = np.arange(np.prod(input_shape)).reshape(input_shape)

    node = ov.cum_sum(input_data, cumsum_axis, reverse=reverse)
    assert node.get_output_size() == 1
    assert node.get_type_name() == "CumSum"
    assert list(node.get_output_shape(0)) == input_shape


def test_normalize_l2():
    input_shape = [1, 2, 3, 4]
    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)
    input_data += 1
    axes = np.array([1, 2, 3]).astype(np.int64)
    eps = 1e-6
    eps_mode = "add"

    node = ov.normalize_l2(input_data, axes, eps, eps_mode)
    assert node.get_output_size() == 1
    assert node.get_type_name() == "NormalizeL2"
    assert list(node.get_output_shape(0)) == input_shape


def test_reduce_with_keywork():
    const = ov.constant([-1], np.int64)
    min_op = ov.reduce_min(node=const, reduction_axes=0)
    assert min_op.get_output_size() == 1
