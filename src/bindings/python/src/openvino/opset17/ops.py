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
def grouped_matmul(
    mat_a: NodeInput,
    mat_b: NodeInput,
    offsets: Optional[NodeInput] = None,
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs Grouped Matrix Multiplication for Mixture of Experts (MoE).

    Computes multiple matrix multiplications where each group processes a subset of the input
    data. Two input combinations are supported:

    - Case 1 (2D x 3D), MoE forward pass, requires ``offsets``:
        - mat_a: (total_tokens, K) - rows partitioned by offsets
        - mat_b: (G, N, K) - per-group weights
        - output: (total_tokens, N)
    - Case 2 (3D x 3D), batched uniform, no offsets:
        - mat_a: (G, M, K) - per-group inputs
        - mat_b: (G, N, K) - per-group weights
        - output: (G, M, N)

    :param mat_a: The first input tensor.
    :param mat_b: The second input tensor with per-group weights.
    :param offsets: 1D tensor of cumulative end-offsets of shape (G,) indicating
                    group boundaries. Required for the 2D x 3D case.
    :param name: The optional name for the new output node.

    :return: The new node performing GroupedMatMul operation.
    """
    inputs = [mat_a, mat_b]
    if offsets is not None:
        inputs.append(offsets)
    return _get_node_factory_opset17().create(
        "GroupedMatMul",
        as_nodes(*inputs, name=name),
    )
