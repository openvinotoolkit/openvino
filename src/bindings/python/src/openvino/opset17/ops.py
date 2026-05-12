# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Factory functions for ops added to openvino opset17."""
from functools import partial
from typing import Optional

from openvino import Node
from openvino.utils.decorators import nameable_op
from openvino.utils.node_factory import _get_node_factory
from openvino.utils.types import NodeInput, as_nodes

_get_node_factory_opset17 = partial(_get_node_factory, "opset17")

# -------------------------------------------- ops ------------------------------------------------


@nameable_op
def atan2(
    y: NodeInput,
    x: NodeInput,
    auto_broadcast: str = "NUMPY",
    name: Optional[str] = None,
) -> Node:
    """Return node performing element-wise Atan2 operation on two input tensors.

    :param y: The node providing the y (ordinate) input tensor.
    :param x: The node providing the x (abscissa) input tensor.
    :param auto_broadcast: Specifies rules used for auto-broadcasting of input tensors.
    :param name: Optional name for output node.
    :return: The node performing element-wise Atan2 operation.
    """
    return _get_node_factory_opset17().create(
        "Atan2",
        as_nodes(y, x, name=name),
        {"auto_broadcast": auto_broadcast.upper()},
    )
