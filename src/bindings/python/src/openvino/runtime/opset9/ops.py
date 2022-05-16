# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Factory functions for all ngraph ops."""
from functools import partial
from typing import Optional

import numpy as np
from openvino.runtime import Node
from openvino.runtime.opset_utils import _get_node_factory
from openvino.runtime.utils.decorators import nameable_op
from openvino.runtime.utils.types import (
    NodeInput,
    as_nodes,
    as_node,
    make_constant_node,
)

_get_node_factory_opset9 = partial(_get_node_factory, "opset9")


# -------------------------------------------- ops ------------------------------------------------


@nameable_op
def eye(
        num_rows: NodeInput,
        num_columns: NodeInput,
        diagonal_index: NodeInput,
        output_type: str,
        batch_shape: Optional[NodeInput] = None,
        name: Optional[str] = None,
) -> Node:
    """Return a node which performs eye operation.

    :param num_rows: The node providing row number tensor.
    :param num_columns: The node providing column number tensor.
    :param diagonal_index: The node providing the index of the diagonal to be populated.
    :param output_type: Specifies the output tensor type, supports any numeric types.
    :param batch_shape: The node providing the leading batch dimensions of output shape. Optionally.
    :param name: The optional new name for output node.
    :return: New node performing deformable convolution operation.
    """
    if batch_shape is not None:
        inputs = as_nodes(num_rows, num_columns, diagonal_index, batch_shape)
    else:
        inputs = as_nodes(num_rows, num_columns, diagonal_index)

    return _get_node_factory_opset9().create("Eye", inputs, {"output_type": output_type})


@nameable_op
def non_max_suppression(
    boxes: NodeInput,
    scores: NodeInput,
    max_output_boxes_per_class: Optional[NodeInput] = None,
    iou_threshold: Optional[NodeInput] = None,
    score_threshold: Optional[NodeInput] = None,
    soft_nms_sigma: Optional[NodeInput] = None,
    box_encoding: str = "corner",
    sort_result_descending: bool = True,
    output_type: str = "i64",
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs NonMaxSuppression.

    :param boxes: Tensor with box coordinates.
    :param scores: Tensor with box scores.
    :param max_output_boxes_per_class: Tensor Specifying maximum number of boxes
                                        to be selected per class.
    :param iou_threshold: Tensor specifying intersection over union threshold
    :param score_threshold: Tensor specifying minimum score to consider box for the processing.
    :param soft_nms_sigma: Tensor specifying the sigma parameter for Soft-NMS.
    :param box_encoding: Format of boxes data encoding.
    :param sort_result_descending: Flag that specifies whenever it is necessary to sort selected
                                   boxes across batches or not.
    :param output_type: Output element type.
    :return: The new node which performs NonMaxSuppression
    """
    max_output_boxes_per_class = max_output_boxes_per_class if max_output_boxes_per_class is not None else make_constant_node(0, np.int64)
    iou_threshold = iou_threshold if iou_threshold is not None else make_constant_node(0, np.float32)
    score_threshold = score_threshold if score_threshold is not None else make_constant_node(0, np.float32)
    soft_nms_sigma = soft_nms_sigma if soft_nms_sigma is not None else make_constant_node(0, np.float32)

    inputs = as_nodes(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, soft_nms_sigma)

    attributes = {
        "box_encoding": box_encoding,
        "sort_result_descending": sort_result_descending,
        "output_type": output_type,
    }

    return _get_node_factory_opset9().create("NonMaxSuppression", inputs, attributes)


def softsign(node: NodeInput, name: Optional[str] = None) -> Node:
    """Apply SoftSign operation on the input node element-wise.

    :param node: One of: input node, array or scalar.
    :param name: The optional name for the output node.
    :return: New node with SoftSign operation applied on each element of it.
    """
    return _get_node_factory_opset9().create("SoftSign", [as_node(node)], {})
