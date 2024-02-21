# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Factory functions for ops added to openvino opset14."""
from functools import partial
from typing import Literal, Optional, Union
import logging

import numpy as np

log = logging.getLogger(__name__)

from openvino.runtime import Node
from openvino.runtime.opset_utils import _get_node_factory
from openvino.runtime.utils.decorators import nameable_op
from openvino.runtime.utils.types import (
    NodeInput,
    as_node,
    as_nodes
)

_get_node_factory_opset14 = partial(_get_node_factory, "opset14")


# -------------------------------------------- ops ------------------------------------------------
@nameable_op
def inverse(
    data: NodeInput,
    adjoint: bool = False
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
