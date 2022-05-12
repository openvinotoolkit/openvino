# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from openvino.runtime import PartialShape, Dimension

import openvino.runtime.opset8 as ov
from openvino.runtime.utils.types import make_constant_node
from tests.runtime import get_runtime
from tests.test_ngraph.util import run_op_node


@pytest.mark.parametrize(
    "ng_api_helper, numpy_function, reduction_axes",
    [
        (ov.reduce_max, np.max, np.array([0, 1, 2, 3])),
        (ov.reduce_min, np.min, np.array([0, 1, 2, 3])),
        (ov.reduce_sum, np.sum, np.array([0, 1, 2, 3])),
        (ov.reduce_prod, np.prod, np.array([0, 1, 2, 3])),
        (ov.reduce_max, np.max, np.array([0])),
        (ov.reduce_min, np.min, np.array([0])),
        (ov.reduce_sum, np.sum, np.array([0])),
        (ov.reduce_prod, np.prod, np.array([0])),
        (ov.reduce_max, np.max, np.array([0, 2])),
        (ov.reduce_min, np.min, np.array([0, 2])),
        (ov.reduce_sum, np.sum, np.array([0, 2])),
        (ov.reduce_prod, np.prod, np.array([0, 2])),
    ],
)
def test_reduction_ops(ng_api_helper, numpy_function, reduction_axes):
    shape = [2, 4, 3, 2]
    np.random.seed(133391)
    input_data = np.random.randn(*shape).astype(np.float32)

    expected = numpy_function(input_data, axis=tuple(reduction_axes))
    result = run_op_node([input_data], ng_api_helper, reduction_axes)
    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    "ng_api_helper, numpy_function, reduction_axes",
    [
        (ov.reduce_logical_and, np.logical_and.reduce, np.array([0])),
        (ov.reduce_logical_or, np.logical_or.reduce, np.array([0])),
        (ov.reduce_logical_and, np.logical_and.reduce, np.array([0, 2])),
        (ov.reduce_logical_or, np.logical_or.reduce, np.array([0, 2])),
        (ov.reduce_logical_and, np.logical_and.reduce, np.array([0, 1, 2, 3])),
        (ov.reduce_logical_or, np.logical_or.reduce, np.array([0, 1, 2, 3])),
    ],
)
def test_reduction_logical_ops(ng_api_helper, numpy_function, reduction_axes):
    shape = [2, 4, 3, 2]
    np.random.seed(133391)
    input_data = np.random.randn(*shape).astype(np.bool)

    expected = numpy_function(input_data, axis=tuple(reduction_axes))
    result = run_op_node([input_data], ng_api_helper, reduction_axes)
    assert np.allclose(result, expected)


def test_topk():
    data_shape = [6, 12, 10, 24]
    data_parameter = ov.parameter(data_shape, name="Data", dtype=np.float32)
    K = np.int32(3)
    axis = np.int32(1)
    node = ov.topk(data_parameter, K, axis, "max", "value")
    assert node.get_type_name() == "TopK"
    assert node.get_output_size() == 2
    assert list(node.get_output_shape(0)) == [6, 3, 10, 24]
    assert list(node.get_output_shape(1)) == [6, 3, 10, 24]


@pytest.mark.parametrize(
    "ng_api_helper, numpy_function, reduction_axes",
    [
        (ov.reduce_mean, np.mean, np.array([0, 1, 2, 3])),
        (ov.reduce_mean, np.mean, np.array([0])),
        (ov.reduce_mean, np.mean, np.array([0, 2])),
    ],
)
def test_reduce_mean_op(ng_api_helper, numpy_function, reduction_axes):
    shape = [2, 4, 3, 2]
    np.random.seed(133391)
    input_data = np.random.randn(*shape).astype(np.float32)

    expected = numpy_function(input_data, axis=tuple(reduction_axes))
    result = run_op_node([input_data], ng_api_helper, reduction_axes)
    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    ("boxes_shape", "scores_shape", "max_output_boxes", "expected_shape"),
    [
        ([1, 1000, 4], [1, 1, 1000], [1000], [PartialShape([Dimension(0, 1000), Dimension(3)]), PartialShape([Dimension(0, 1000), Dimension(3)])]),
        ([1, 700, 4], [1, 1, 700], [600], [PartialShape([Dimension(0, 600), Dimension(3)]), PartialShape([Dimension(0, 600), Dimension(3)])]),
        ([1, 300, 4], [1, 1, 300], [300], [PartialShape([Dimension(0, 300), Dimension(3)]), PartialShape([Dimension(0, 300), Dimension(3)])]),
    ],
)
def test_non_max_suppression(boxes_shape, scores_shape, max_output_boxes, expected_shape):
    boxes_parameter = ov.parameter(boxes_shape, name="Boxes", dtype=np.float32)
    scores_parameter = ov.parameter(scores_shape, name="Scores", dtype=np.float32)

    node = ov.non_max_suppression(boxes_parameter, scores_parameter, make_constant_node(max_output_boxes, np.int64))
    assert node.get_type_name() == "NonMaxSuppression"
    assert node.get_output_size() == 3
    assert node.get_output_partial_shape(0) == expected_shape[0]
    assert node.get_output_partial_shape(1) == expected_shape[1]
    assert list(node.get_output_shape(2)) == [1]


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
    expected_shape = [1000, 256, 6, 6]

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
    assert list(node.get_output_shape(0)) == expected_shape


@pytest.mark.parametrize(
    "input_shape, cumsum_axis, reverse",
    [([5, 2], 0, False), ([5, 2], 1, False), ([5, 2, 6], 2, False), ([5, 2], 0, True)],
)
def test_cum_sum(input_shape, cumsum_axis, reverse):
    input_data = np.arange(np.prod(input_shape)).reshape(input_shape)

    if reverse:
        expected = np.cumsum(input_data[::-1], axis=cumsum_axis)[::-1]
    else:
        expected = np.cumsum(input_data, axis=cumsum_axis)

    runtime = get_runtime()
    node = ov.cum_sum(input_data, cumsum_axis, reverse=reverse)
    computation = runtime.computation(node)
    result = computation()
    assert np.allclose(result, expected)


def test_normalize_l2():
    input_shape = [1, 2, 3, 4]
    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)
    input_data += 1
    axes = np.array([1, 2, 3]).astype(np.int64)
    eps = 1e-6
    eps_mode = "add"

    runtime = get_runtime()
    node = ov.normalize_l2(input_data, axes, eps, eps_mode)
    computation = runtime.computation(node)
    result = computation()

    expected = np.array(
        [
            0.01428571,
            0.02857143,
            0.04285714,
            0.05714286,
            0.07142857,
            0.08571429,
            0.1,
            0.11428571,
            0.12857144,
            0.14285715,
            0.15714286,
            0.17142858,
            0.18571429,
            0.2,
            0.21428572,
            0.22857143,
            0.24285714,
            0.25714287,
            0.27142859,
            0.2857143,
            0.30000001,
            0.31428573,
            0.32857144,
            0.34285715,
        ]
    ).reshape(input_shape)

    assert np.allclose(result, expected)
