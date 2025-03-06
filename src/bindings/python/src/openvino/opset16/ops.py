# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Factory functions for ops added to openvino opset16."""
from functools import partial
from typing import Optional, Literal

from openvino import Node
from openvino.utils.decorators import nameable_op
from openvino.utils.node_factory import _get_node_factory
from openvino.utils.types import NodeInput, as_nodes

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
