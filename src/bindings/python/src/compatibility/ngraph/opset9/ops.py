# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Factory functions for all ngraph ops."""
from functools import partial
from typing import Optional

import numpy as np
from ngraph.impl import Node
from ngraph.opset_utils import _get_node_factory
from ngraph.utils.decorators import nameable_op
from ngraph.utils.types import (
    NodeInput,
    as_nodes,
    as_node
)

_get_node_factory_opset9 = partial(_get_node_factory, "opset9")


# -------------------------------------------- ops ------------------------------------------------


@nameable_op
def eye(
        num_rows: NodeInput,
        num_columns: NodeInput,
        diagonal_index: NodeInput,
        output_type: str,
        batch_shape: Optional[NodeInput] = None,
        name: Optional[str] = None,
) -> Node:
    """Return a node which performs eye operation.

    :param num_rows: The node providing row number tensor.
    :param num_columns: The node providing column number tensor.
    :param diagonal_index: The node providing the index of the diagonal to be populated.
    :param output_type: Specifies the output tensor type, supports any numeric types.
    :param batch_shape: The node providing the leading batch dimensions of output shape. Optionally.
    :param name: The optional new name for output node.
    :return: New node performing deformable convolution operation.
    """
    if batch_shape is not None:
        inputs = as_nodes(num_rows, num_columns, diagonal_index, batch_shape)
    else:
        inputs = as_nodes(num_rows, num_columns, diagonal_index)

    return _get_node_factory_opset9().create("Eye", inputs, {"output_type": output_type})


@nameable_op
def softsign(node: NodeInput, name: Optional[str] = None) -> Node:
    """Apply SoftSign operation on the input node element-wise.

    :param node: One of: input node, array or scalar.
    :param name: The optional name for the output node.
    :return: New node with SoftSign operation applied on each element of it.
    """
    return _get_node_factory_opset9().create("SoftSign", [as_node(node)], {})
