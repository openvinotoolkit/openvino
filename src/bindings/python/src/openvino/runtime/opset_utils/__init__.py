# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.runtime import Node
from openvino.utils.decorators import nameable_op
from openvino.utils.node_factory import NodeFactory
from openvino.utils.types import (
    as_node,
    NodeInput,
)

from openvino.utils.node_factory import _get_node_factory
