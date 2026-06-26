# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Factory functions for ops added to openvino opset17."""
from functools import partial
from typing import Optional

from openvino import Node
from openvino.utils.decorators import nameable_op, unary_op
from openvino.utils.node_factory import _get_node_factory
from openvino.utils.types import NodeInput, as_nodes

_get_node_factory_opset17 = partial(_get_node_factory, "opset17")

# -------------------------------------------- ops ------------------------------------------------


@unary_op
def erfinv(node: NodeInput, name: Optional[str] = None) -> Node:
    """Return node which calculates the inverse error function element-wise on the input tensor.

    :param node: The node providing data for the operation. Must be of floating-point type.
    :param name: The optional name for the new output node.
    :return: The new node performing the element-wise ErfInv operation.
    """
    return _get_node_factory_opset17().create("ErfInv", [node])


@nameable_op
def bincount(
    data: NodeInput,
    weights: Optional[NodeInput] = None,
    minlength: int = 0,
    name: Optional[str] = None,
) -> Node:
    """Count occurrences of each value in a 1-D tensor of non-negative integers.

    :param data: 1-D non-negative integer tensor.
    :param weights: Optional 1-D float/integer tensor, same length as data.
    :param minlength: Minimum length of the output tensor; defaults to 0.
    :param name: The optional name for the new output node.
    :return: The new Bincount node.
    """
    inputs = [data]
    if weights is not None:
        inputs.append(weights)
    return _get_node_factory_opset17().create("Bincount", as_nodes(*inputs, name=name), {"minlength": minlength})
