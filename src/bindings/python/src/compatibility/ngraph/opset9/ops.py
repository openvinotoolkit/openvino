# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Factory functions for all ngraph ops."""
from functools import partial
from typing import Optional

from ngraph.impl import Node
from ngraph.opset_utils import _get_node_factory
from ngraph.utils.decorators import nameable_op
from ngraph.utils.types import NodeInput, as_node


_get_node_factory_opset9 = partial(_get_node_factory, "opset9")


# -------------------------------------------- ops ------------------------------------------------

@nameable_op
def softsign(data: NodeInput, name: Optional[str] = None) -> Node:
    """Apply SoftSign operation on each element of input tensor.

    :param data: The tensor providing input data.
    :param name: The optional name for the output node
    :return: The new node with SoftSign operation applied on each element.
    """
    return _get_node_factory_opset9().create("SoftSign", [as_node(data)], {})
