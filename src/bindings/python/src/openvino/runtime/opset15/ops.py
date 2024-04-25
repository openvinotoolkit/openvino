# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Factory functions for ops added to openvino opset15."""
from functools import partial
from typing import Optional, Literal

from openvino.runtime import Node, Type
from openvino.runtime.opset_utils import _get_node_factory
from openvino.runtime.utils.decorators import nameable_op
from openvino.runtime.utils.types import NodeInput, as_nodes

_get_node_factory_opset15 = partial(_get_node_factory, "opset15")

# -------------------------------------------- ops ------------------------------------------------


@nameable_op
def scatter_nd_update(
    data: NodeInput,
    indices: NodeInput,
    updates: NodeInput,
    reduction: Optional[Literal["none", "sum", "sub", "prod", "min", "max"]] = None,
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs ScatterNDUpdate.

    :param data: Node input representing the tensor to be updated.
    :param indices: Node input representing the indices at which updates will be applied.
    :param updates: Node input representing the updates to be applied.
    :param reduction: The type of operation to perform on the inputs. One of "none", "sum",
                      "sub", "prod", "min", "max".
    :param name: Optional name for the output node.
    :return: New node performing the ScatterNDUpdate.
    """
    inputs = as_nodes(data, indices, updates, name=name)
    attributes = {}
    if reduction:
        attributes["reduction"] = reduction
    return _get_node_factory_opset15().create("ScatterNDUpdate", inputs, attributes)
