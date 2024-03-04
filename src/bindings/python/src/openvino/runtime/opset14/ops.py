# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Factory functions for ops added to openvino opset14."""
from functools import partial
from typing import Union

from openvino.runtime import Node, Type
from openvino.runtime.opset_utils import _get_node_factory
from openvino.runtime.utils.decorators import nameable_op
from openvino.runtime.utils.types import NodeInput, as_nodes

_get_node_factory_opset14 = partial(_get_node_factory, "opset14")


# -------------------------------------------- ops ------------------------------------------------
@nameable_op
def convert_promote_types(
    left_node: NodeInput,
    right_node: NodeInput,
    promote_unsafe: bool = False,
    pytorch_scalar_promotion: bool = False,
    u64_integer_promotion_target: Union[str, Type] = "f32",
) -> Node:
    """Return a node performing conversion to common type based on promotion rules.

    :param left_node: input node with type to be promoted to common one.
    :param right_node: input node with type to be promoted to common one.
    :param promote_unsafe: Bool attribute whether to allow promotions that might result in bit-widening, precision loss and undefined behaviors.
    :param pytorch_scalar_promotion: Bool attribute whether to promote scalar input to type provided by non-scalar input when number format is matching.
    :param u64_integer_promotion_target: Element type attribute to select promotion result when inputs are u64 and signed integer.

    :return: The new node performing ConvertPromoteTypes operation.
    """
    inputs = as_nodes(left_node, right_node)

    attributes = {
        "promote_unsafe": promote_unsafe,
        "pytorch_scalar_promotion": pytorch_scalar_promotion,
        "u64_integer_promotion_target": u64_integer_promotion_target,
    }
    return _get_node_factory_opset14().create("ConvertPromoteTypes", inputs, attributes)


@nameable_op
def inverse(
    data: NodeInput,
    adjoint: bool = False,
) -> Node:
    """Return a node with inverse matrices of the input.

    :param data: Tensor with matrices to invert. Last two dimensions must be of the same size.
    :param adjoint: Whether to return adjoint instead of inverse matrices. Defaults to false.

    :return: The new node performing Inverse operation.
    """
    inputs = as_nodes(data)

    attributes = {
        "adjoint": adjoint,
    }

    return _get_node_factory_opset14().create("Inverse", inputs, attributes)
