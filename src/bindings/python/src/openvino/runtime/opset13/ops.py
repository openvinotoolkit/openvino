# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Factory functions for ops added to openvino opset13."""
from functools import partial
from typing import Literal, Optional

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
@binary_op
def bitwise_and(
    left_node: NodeInput,
    right_node: NodeInput,
    auto_broadcast: str = "NUMPY",
    name: Optional[str] = None,
) -> Node:
    """Return node which performs bitwise AND operation on input nodes element-wise.

    For boolean input tensors, operator is equivalent to logical_and.

    :param left_node: Tensor of integer or boolean datatype providing data.
    :param right_node: Tensor of integer or boolean datatype providing data.
    :param auto_broadcast: The type of broadcasting specifies rules used for auto-broadcasting of input tensors. Defaults to “NUMPY”.
    :param name: The optional new name for output node.
    :return: The node performing bitwise AND operation on input nodes corresponding elements.
    """
    return _get_node_factory_opset13().create(
        "BitwiseAnd",
        [left_node, right_node],
        {"auto_broadcast": auto_broadcast.upper()},
    )


@unary_op
def bitwise_not(
    node: NodeInput,
    name: Optional[str] = None,
) -> Node:
    """Return node which performs bitwise NOT operation on input node element-wise.

    For boolean input tensors, operator is equivalent to logical_not.

    :param node: Tensor of integer or boolean datatype providing data.
    :param name: The optional new name for output node.
    :return: The node performing bitwise NOT operation on the given tensor.
    """
    return _get_node_factory_opset13().create(
        "BitwiseNot",
        [node],
    )


@binary_op
def bitwise_or(
    left_node: NodeInput,
    right_node: NodeInput,
    auto_broadcast: str = "NUMPY",
    name: Optional[str] = None,
) -> Node:
    """Return node which performs bitwise OR operation on input nodes element-wise.

    For boolean input tensors, operator is equivalent to logical_or.

    :param left_node: Tensor of integer or boolean datatype providing data.
    :param right_node: Tensor of integer or boolean datatype providing data.
    :param auto_broadcast: The type of broadcasting specifies rules used for auto-broadcasting of input tensors. Defaults to “NUMPY”.
    :param name: The optional new name for output node.
    :return: The node performing bitwise OR operation on input nodes corresponding elements.
    """
    return _get_node_factory_opset13().create(
        "BitwiseOr",
        [left_node, right_node],
        {"auto_broadcast": auto_broadcast.upper()},
    )


@binary_op
def bitwise_xor(
    left_node: NodeInput,
    right_node: NodeInput,
    auto_broadcast: str = "NUMPY",
    name: Optional[str] = None,
) -> Node:
    """Return node which performs bitwise XOR operation on input nodes element-wise.

    For boolean input tensors, operator is equivalent to logical_xor.

    :param left_node: Tensor of integer or boolean datatype providing data.
    :param right_node: Tensor of integer or boolean datatype providing data.
    :param auto_broadcast: The type of broadcasting specifies rules used for auto-broadcasting of input tensors. Defaults to “NUMPY”.
    :param name: The optional new name for output node.
    :return: The node performing bitwise XOR operation on input nodes corresponding elements.
    """
    return _get_node_factory_opset13().create(
        "BitwiseXor",
        [left_node, right_node],
        {"auto_broadcast": auto_broadcast.upper()},
    )


@nameable_op
def fake_convert(
    data: NodeInput,
    scale: NodeInput,
    shift: Optional[NodeInput] = None,
    destination_type: Literal["f8e4m3", "f8e5m2"] = "f8e4m3",
    name: Optional[str] = None,
) -> Node:
    r"""FakeConvert is element-wise emulation of float8 type on the original type of the data input.

    :param data: The node with data tensor with FP16 or FP32 datatype.
    :param scale: Tensor with a scale factor for the data input value,
                  with datatype of FP16 or FP32 and shape Numpy-broadcastable to data.
    :param shift: Optional tensor with value to subtract before and add after conversion of the data input value,
                  with datatype of FP16 or FP32 and shape Numpy-broadcastable to data.
    :param destination_type: Type to emulate, string of either "f8e4m3" or "f8e5m2".
    :param name: The optional new name for output node.

    :return: The new node performing FakeConvert operation.
    """
    nodes = [data, scale]
    if shift is not None:
        nodes.append(shift)
    return _get_node_factory_opset13().create(
        "FakeConvert",
        as_nodes(*nodes),
        {"destination_type": destination_type.lower()},
    )


@nameable_op
def multinomial(
    probs: NodeInput,
    num_samples: NodeInput,
    convert_type: str,
    with_replacement: bool,
    log_probs: bool,
    global_seed: int = 0,
    op_seed: int = 0,
) -> Node:
    """Return a node which generates a sequence of class indices sampled from the multinomial distribution.

    :param probs: Tensor with probabilities of floating-point type, and shape [class_size] or [batch_size, class_size].
    :param num_samples: Tensor (scalar or 1D) a single element of type i32 or i64,
                        specifying the number of samples to draw from the multinomial distribution.
    :param convert_type: Specifies the output tensor type, possible values: 'i64', 'i32'.
    :param with_replacement: Flag that specifies whether to sample with replacement.
    :param log_probs: Flag that specifies whether *probs* should be treated as unnormalized log probabilities.
    :param global_seed: Specifies global seed value. Required to be a positive integer or 0.
    :param op_seed: Specifies operational seed value. Required to be a positive integer or 0.

    :return: The new node performing Multinomial operation.
    """
    inputs = as_nodes(probs, num_samples)

    if global_seed < 0:
        raise RuntimeError(
            f"global_seed should be positive or 0. Got: {global_seed}")

    if op_seed < 0:
        raise RuntimeError(f"op_seed should be positive or 0. Got: {op_seed}")

    attributes = {
        "convert_type": convert_type,
        "with_replacement": with_replacement,
        "log_probs": log_probs,
        "global_seed": global_seed,
        "op_seed": op_seed,
    }
    return _get_node_factory_opset13().create("Multinomial", inputs, attributes)


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
    inputs = as_nodes(boxes, scores, max_output_boxes_per_class,
                      iou_threshold, score_threshold)

    attributes = {
        "sort_result_descending": sort_result_descending,
        "output_type": output_type,
        "clockwise": clockwise,
    }

    return _get_node_factory_opset13().create("NMSRotated", inputs, attributes)


@nameable_op
def scaled_dot_product_attention(
    query: NodeInput,
    key: NodeInput,
    value: NodeInput,
    attention_mask: Optional[NodeInput] = None,
    scale: Optional[NodeInput] = None,
    causal: bool = False,
    name: Optional[str] = None,
) -> Node:
    """Return a node which implements Scaled Dot Product Attention.

    :param query: Query tensor of shape [N, ..., L, E] and floating-point datatype.
    :param key: Key tensor of shape [N, ..., S, E] and floating-point datatype.
    :param value: Value tensor of shape [N, ..., S, Ev] and floating-point datatype.
    :param attention_mask: Optional attention mask tensor of shape [N, ..., L, S] or scalar float type zero value.
                           Refer to the operation specification for a complete description.
    :param scale: Optional alternative scale, a floating-point type scalar.
    :param causal: If true, then autogenerates causal attention mask instead of using attention_mask input.
                   In this case attention_mask input is ignored.
    :param name: The optional new name for output node.

    :return: The new node performing Scaled Dot Product Attention operation.
    """
    inputs = as_nodes(query, key, value, attention_mask) if attention_mask is not None else as_nodes(
        query, key, value, scale)

    attributes = {
        "causal": causal,
    }
    return _get_node_factory_opset13().create("ScaledDotProductAttention", inputs, attributes)
