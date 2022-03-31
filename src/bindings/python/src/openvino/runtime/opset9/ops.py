# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Factory functions for all openvino ops."""
from functools import partial
from typing import Optional
from openvino.runtime import Node
from openvino.runtime.opset_utils import _get_node_factory
from openvino.runtime.utils.decorators import nameable_op
from openvino.runtime.utils.types import NodeInput, as_node


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
