# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import openvino.opset13 as ov_opset13

from openvino import PartialShape, Dimension, Type
from openvino.utils.types import make_constant_node


@pytest.mark.parametrize(
    ("boxes_shape", "scores_shape", "max_output_boxes", "iou_threshold", "score_threshold", "expected_shape"),
    [
        ([1, 100, 5], [1, 1, 100], [100], 0.1, 0.4, [PartialShape([Dimension(0, 100), Dimension(3)]), PartialShape([Dimension(0, 100), Dimension(3)])]),
        ([1, 700, 5], [1, 1, 700], [600], 0.1, 0.4, [PartialShape([Dimension(0, 600), Dimension(3)]), PartialShape([Dimension(0, 600), Dimension(3)])]),
        ([1, 300, 5], [1, 1, 300], [300], 0.1, 0.4, [PartialShape([Dimension(0, 300), Dimension(3)]), PartialShape([Dimension(0, 300), Dimension(3)])]),
    ],
)
@pytest.mark.parametrize("op_name", ["nms_rotated", "nmsRotated", "nmsRotatedOpset13"])
def test_nms_rotated_default_attrs(boxes_shape, scores_shape, max_output_boxes, iou_threshold, score_threshold, expected_shape, op_name):
    boxes_parameter = ov_opset13.parameter(boxes_shape, name="Boxes", dtype=np.float32)
    scores_parameter = ov_opset13.parameter(scores_shape, name="Scores", dtype=np.float32)

    max_output_boxes = make_constant_node(max_output_boxes, np.int64)
    iou_threshold = make_constant_node(iou_threshold, np.float32)
    score_threshold = make_constant_node(score_threshold, np.float32)

    node = ov_opset13.nms_rotated(boxes_parameter, scores_parameter, max_output_boxes,
                                  iou_threshold, score_threshold, name=op_name)
    assert node.get_type_name() == "NMSRotated"
    assert node.get_friendly_name() == op_name
    assert node.get_output_size() == 3
    assert node.get_output_partial_shape(0) == expected_shape[0]
    assert node.get_output_partial_shape(1) == expected_shape[1]
    assert node.get_output_partial_shape(2) == PartialShape([1])


@pytest.mark.parametrize(
    ("boxes_shape", "scores_shape", "max_output_boxes", "iou_threshold", "score_threshold",
     "sort_result_descending", "output_type", "clockwise", "expected_shape"),
    [
        ([1, 100, 5], [1, 1, 100], [100], 0.1, 0.4, False, "i64", False,
         [PartialShape([Dimension(0, 100), Dimension(3)]), PartialShape([Dimension(0, 100), Dimension(3)])]),
        ([1, 100, 5], [1, 1, 100], [100], 0.1, 0.4, True, "i32", True,
         [PartialShape([Dimension(0, 100), Dimension(3)]), PartialShape([Dimension(0, 100), Dimension(3)])]),
    ],
)
@pytest.mark.parametrize("op_name", ["nms_rotated", "nmsRotated", "nmsRotatedOpset13", "nmsRotatedcustomAttrs"])
def test_nms_rotated_custom_attrs(boxes_shape, scores_shape, max_output_boxes, iou_threshold, score_threshold,
                                  sort_result_descending, output_type, clockwise, expected_shape, op_name):
    boxes_parameter = ov_opset13.parameter(boxes_shape, name="Boxes", dtype=np.float32)
    scores_parameter = ov_opset13.parameter(scores_shape, name="Scores", dtype=np.float32)

    max_output_boxes = make_constant_node(max_output_boxes, np.int64)
    iou_threshold = make_constant_node(iou_threshold, np.float32)
    score_threshold = make_constant_node(score_threshold, np.float32)

    node = ov_opset13.nms_rotated(boxes_parameter, scores_parameter, max_output_boxes, iou_threshold,
                                  score_threshold, sort_result_descending, output_type, clockwise, name=op_name)
    assert node.get_type_name() == "NMSRotated"
    assert node.get_friendly_name() == op_name
    assert node.get_output_size() == 3
    assert node.get_output_partial_shape(0) == expected_shape[0]
    assert node.get_output_partial_shape(1) == expected_shape[1]
    assert node.get_output_partial_shape(2) == PartialShape([1])

    assert node.get_output_element_type(0) == Type.i32 if output_type == "i32" else Type.i64
    assert node.get_output_element_type(1) == Type.f32
    assert node.get_output_element_type(2) == Type.i32 if output_type == "i32" else Type.i64
