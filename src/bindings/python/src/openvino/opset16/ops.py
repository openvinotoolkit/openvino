# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Factory functions for ops added to openvino opset16."""
from functools import partial
from typing import Optional, Literal

from openvino import Node
from openvino.utils.decorators import nameable_op
from openvino.utils.node_factory import _get_node_factory
from openvino.utils.types import NodeInput, as_nodes, as_node, TensorShape

_get_node_factory_opset16 = partial(_get_node_factory, "opset16")

# -------------------------------------------- ops ------------------------------------------------


@nameable_op
def identity(
    data: NodeInput,
    name: Optional[str] = None,
) -> Node:
    """Identity operation is used as a placeholder. It creates a copy of the input to forward to the output.

    :param data: Tensor with data.

    :return: The new node performing Identity operation.
    """
    return _get_node_factory_opset16().create(
        "Identity",
        as_nodes(data, name=name),
        {},
    )


@nameable_op
def istft(
    data: NodeInput,
    window: NodeInput,
    frame_size: NodeInput,
    frame_step: NodeInput,
    center: bool,
    normalized: bool,
    signal_length: Optional[NodeInput] = None,
    name: Optional[str] = None,
) -> Node:
    """Return a node which generates ISTFT operation.

    :param  data: The node providing input data.
    :param  window: The node providing window data.
    :param  frame_size: The node with scalar value representing the size of Fourier Transform.
    :param  frame_step: The distance (number of samples) between successive window frames.
    :param  center: Flag signaling if the signal input has been padded before STFT.
    :param  normalized: Flag signaling if the STFT result has been normalized.
    :param  signal_length: The optional node with length of the original signal.
    :param  name: The optional name for the created output node.
    :return: The new node performing ISTFT operation.
    """
    if signal_length is None:
        inputs = as_nodes(data, window, frame_size, frame_step, name=name)
    else:
        inputs = as_nodes(data, window, frame_size, frame_step, signal_length, name=name)
    return _get_node_factory_opset16().create(
        "ISTFT",
        inputs,
        {"center": center, "normalized": normalized},
    )


@nameable_op
def segment_max(
    data: NodeInput,
    segment_ids: NodeInput,
    num_segments: Optional[NodeInput] = None,
    fill_mode: Optional[Literal["ZERO", "LOWEST"]] = None,
    name: Optional[str] = None,
) -> Node:
    """The SegmentMax operation finds the maximum value in each specified segment of the input tensor.

    :param data: ND tensor of type T, the numerical data on which SegmentMax operation will be performed.
    :param segment_ids: 1D Tensor of sorted non-negative numbers, representing the segments.
    :param num_segments: An optional scalar value representing the segments count. If not provided, it is inferred from segment_ids.
    :param fill_mode: Responsible for the value assigned to segments which are empty. Can be "ZERO" or "LOWEST".
    :param name: Optional name for the node.

    :return: The new node performing SegmentMax operation.
    """
    if fill_mode is None:
        raise ValueError("fill_mode must be provided and can be either 'ZERO' or 'LOWEST'")
    inputs = [data, segment_ids]
    if num_segments is not None:
        inputs.append(num_segments)
    return _get_node_factory_opset16().create(
        "SegmentMax",
        as_nodes(*inputs, name=name),
        {"fill_mode": fill_mode},
    )


@nameable_op
def sparse_fill_empty_rows(
    values: NodeInput,
    dense_shape: NodeInput,
    indices: NodeInput,
    default_value: NodeInput,
    name: Optional[str] = None,
) -> Node:
    """Fills empty rows of an input sparse tensor with a default value.

    :param values: 1D tensor containing the values to be inserted at the specified indices.
    :param dense_shape: 1D tensor indicating the shape of the 2D dense tensor.
    :param indices: 2D tensor indicating the positions at which values are placed.
    :param default_value: A scalar value to be inserted into empty rows.
    :param name: Optional name for the node.

    :return: The new node performing SparseFillEmptyRows operation with three outputs:
             [output_indices, output_values, empty_row_indicator]
    """
    return _get_node_factory_opset16().create(
        "SparseFillEmptyRows",
        as_nodes(values, dense_shape, indices, default_value, name=name),
        {},
    )


@nameable_op
def avg_pool(
    data_batch: NodeInput,
    strides: list[int],
    pads_begin: TensorShape,
    pads_end: TensorShape,
    kernel_shape: TensorShape,
    exclude_pad: bool,
    rounding_type: str = "floor",
    auto_pad: Optional[str] = None,
    dilations: Optional[list[int]] = None,
    name: Optional[str] = None,
) -> Node:
    """Return average pooling node.

    :param data_batch:      The input node providing data.
    :param strides:         The window movement strides.
    :param pads_begin:      The number of pixels to add at the beginning along each axis.
    :param pads_end:        The number of pixels to add at the end along each axis.
    :param kernel_shape:    The pooling window shape.
    :param exclude_pad:     Whether or not to include zero padding in average computations.
    :param rounding_type:   Determines used rounding schema when computing output shape. Acceptable
                            values are: ['floor', 'ceil', 'ceil_torch']. Defaults to 'floor'.
    :param auto_pad:        Determines how the padding is calculated. Acceptable values:
                            [None, 'same_upper', 'same_lower', 'valid']. Defaults to None.
    :param dilations:       The index of the next pixel to select when pooling. If not provided,
                            defaults to [1, 1, ...] (no dilation).
    :param name:            Optional name for the new output node.

    :return: New node with AvgPool operation applied on its data.
    """
    if auto_pad is None:
        auto_pad = "explicit"

    attributes = {
        "strides": strides,
        "pads_begin": pads_begin,
        "pads_end": pads_end,
        "kernel": kernel_shape,
        "exclude-pad": exclude_pad,
        "rounding_type": rounding_type.upper(),
        "auto_pad": auto_pad.upper(),
    }

    if dilations is not None:
        attributes["dilations"] = dilations

    return _get_node_factory_opset16().create(
        "AvgPool",
        [as_node(data_batch, name=name)],
        attributes,
    )


@nameable_op
def one_hot(
    indices: NodeInput,
    depth: NodeInput,
    on_value: NodeInput,
    off_value: NodeInput,
    axis: int,
    negative_indices_mode: Optional[str] = None,
    name: Optional[str] = None,
) -> Node:
    """Create node performing one-hot encoding on input data.

    :param indices: Input tensor of rank N with indices of any supported integer data type.
    :param depth: Scalar of any supported integer type that specifies number of classes and
                  the size of one-hot dimension.
    :param on_value: Scalar of any type that is the value that the locations
                     in output tensor represented by indices in input take.
    :param off_value: Scalar of any type that is the value that the locations not represented
                      by indices in input take.
    :param axis: New axis position in the output shape to fill with one-hot values.
    :param negative_indices_mode: Controls how negative indices are handled. Can be 'ignore_negative'
                                  (negative indices are ignored and filled with off_value) or
                                  'normalize' (negative indices in range [-depth, -1] are normalized).
                                  If not provided, defaults to 'ignore_negative'.
    :param name: The optional name for new output node.
    :return: New node performing one-hot operation.
    """
    attributes = {"axis": axis}
    if negative_indices_mode is not None:
        attributes["negative_indices_mode"] = negative_indices_mode

    return _get_node_factory_opset16().create(
        "OneHot",
        as_nodes(indices, depth, on_value, off_value, name=name),
        attributes,
    )
