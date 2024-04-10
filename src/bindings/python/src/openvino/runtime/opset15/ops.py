# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Factory functions for ops added to openvino opset14."""
from functools import partial

from typing import Union, Optional, List

from openvino.runtime import Node, Type
from openvino.runtime.opset_utils import _get_node_factory
from openvino.runtime.utils.types import TensorShape
from openvino.runtime.utils.decorators import nameable_op
from openvino.runtime.utils.types import NodeInput, as_node, as_nodes

_get_node_factory_opset14 = partial(_get_node_factory, "opset14")


# -------------------------------------------- ops ------------------------------------------------
