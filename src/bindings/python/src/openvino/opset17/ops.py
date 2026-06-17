# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Factory functions for ops added to openvino opset17."""
from functools import partial
from typing import Optional

from openvino import Node
from openvino.utils.decorators import unary_op
from openvino.utils.node_factory import _get_node_factory
from openvino.utils.types import NodeInput

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
