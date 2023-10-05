# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Factory functions for ops added to openvino opset13."""
from functools import partial
from typing import Optional

from openvino.runtime import Node
from openvino.runtime.opset_utils import _get_node_factory
from openvino.runtime.utils.decorators import binary_op, nameable_op, unary_op
from openvino.runtime.utils.types import (
    NodeInput,
    as_nodes,
    as_node,
)

_get_node_factory_opset13 = partial(_get_node_factory, "opset13")


# -------------------------------------------- ops ------------------------------------------------
@unary_op
def bitwise_not(
    node: NodeInput,
    name: Optional[str] = None,
) -> Node:
    """Return node which performs bitwise NOT operation on input node element-wise.

    For boolean input tensors, operator is equivalent to logical_not.

    :param node: Tensor of integer or boolean datatype providing data.
    :param name: The optional new name for output node.
    :return: The node performing bitwise NOT operation with given tensor.
    """
    return _get_node_factory_opset13().create(
        "BitwiseNot",
        [node],
    )


@nameable_op
def nms_rotated(
    boxes: NodeInput,
    scores: NodeInput,
    max_output_boxes_per_class: NodeInput,
    iou_threshold: NodeInput,
    score_threshold: NodeInput,
    sort_result_descending: bool = True,
    output_type: str = "i64",
    clockwise: bool = True,
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs NMSRotated.

    :param boxes: Tensor with box coordinates of floating point type and shape [num_batches, num_boxes, 5],
                  where the last dimension is defined as [x_ctr, y_ctr, width, height, angle_radians].
    :param scores: Tensor with box scores of floating point type and shape [num_batches, num_classes, num_boxes].
    :param max_output_boxes_per_class: Tensor (scalar or 1D) of integer type, specifying maximum number of boxes
                                        to be selected per class.
    :param iou_threshold: Tensor (scalar or 1D) of floating point type, specifying intersection over union threshold
    :param score_threshold: Tensor (scalar or 1D) of floating point type, specifying minimum score to consider box for the processing.
    :param sort_result_descending: Flag that specifies whenever it is necessary to sort selected
                                   boxes across batches or not.
    :param output_type: Output element type.
    :param clockwise: Flag that specifies direction of the box rotation.
    :return: The new node which performs NMSRotated
    """
    inputs = as_nodes(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)

    attributes = {
        "sort_result_descending": sort_result_descending,
        "output_type": output_type,
        "clockwise": clockwise,
    }

    return _get_node_factory_opset13().create("NMSRotated", inputs, attributes)
