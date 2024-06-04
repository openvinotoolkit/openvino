# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Factory functions for ops added to openvino opset15."""
from functools import partial
from typing import Optional, Literal, List

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


@nameable_op
def col2im(
    data: NodeInput,
    output_size: NodeInput,
    kernel_size: NodeInput,
    strides: Optional[List[int]] = None,
    dilations: Optional[List[int]] = None,
    pads_begin: Optional[List[int]] = None,
    pads_end: Optional[List[int]] = None,
    name: Optional[str] = None,
) -> Node:
    """Perform data movement operation which combines sliding blocks into an image tensor.

    :param  data:                The node providing input data.
    :param  output_size:         Shape of the spatial dimensions of the output image.
    :param  kernel_size:         Size of the sliding blocks.
    :param  strides:             Stride on the sliding blocks in the input spatial dimensions. Defaults to [1, 1].
    :param  dilations:           The dilation of filter elements (distance between elements). Defaults to [1, 1].
    :param  pads_begin:          The number of pixels added at the beginning along each axis. Defaults to [0, 0].
    :param  pads_end:            The number of pixels added at the end along each axis. Defaults to [0, 0].
    :param  name:                The optional name for the created output node.

    :return:   The new node performing Col2Im operation.
    """
    if strides is None:
        strides = [1, 1]
    if dilations is None:
        dilations = [1, 1]
    if pads_begin is None:
        pads_begin = [0, 0]
    if pads_end is None:
        pads_end = [0, 0]
    return _get_node_factory_opset15().create(
        "Col2Im",
        as_nodes(data, output_size, kernel_size, name=name),
        {
            "strides": strides,
            "dilations": dilations,
            "pads_begin": pads_begin,
            "pads_end": pads_end,
        },
    )
