# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Factory functions for all ngraph ops."""
from functools import partial
from typing import Optional

from ngraph.impl import Node
from ngraph.opset_utils import _get_node_factory
from ngraph.utils.decorators import nameable_op
from ngraph.utils.types import (
    NodeInput,
    as_nodes,
    as_node
)

_get_node_factory_opset12 = partial(_get_node_factory, "opset12")


# -------------------------------------------- ops ------------------------------------------------

@nameable_op
def pad(
    arg: NodeInput,
    pads_begin: NodeInput,
    pads_end: NodeInput,
    pad_mode: str,
    arg_pad_value: Optional[NodeInput] = None,
    name: Optional[str] = None,
) -> Node:
    """Return a generic padding operation.

    :param arg: The node producing input tensor to be padded.
    :param pads_begin: number of padding elements to be added before position 0
                       on each axis of arg. Negative indices are supported.
    :param pads_end: number of padding elements to be added after the last element.
                     Negative indices are supported.
    :param pad_mode: "constant", "edge", "reflect" or "symmetric"
    :param arg_pad_value: value used for padding if pad_mode is "constant"
    :return: Pad operation node.
    """
    input_nodes = as_nodes(arg, pads_begin, pads_end)
    if arg_pad_value:
        input_nodes.append(as_node(arg_pad_value))

    pad_mode = pad_mode.upper()
    return _get_node_factory_opset12().create("Pad", input_nodes, {"pad_mode": pad_mode})
