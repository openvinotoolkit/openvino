# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
import numpy as np

from openvino.runtime import Node
from openvino.runtime.utils.decorators import nameable_op
from openvino.runtime.utils.node_factory import NodeFactory
from openvino.runtime.utils.types import (
    as_node,
    NodeInput,
)


def _get_node_factory(opset_version: Optional[str] = None) -> NodeFactory:
    """Return NodeFactory configured to create operators from specified opset version."""
    if opset_version:
        return NodeFactory(opset_version)
    else:
        return NodeFactory()
