# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.ops.unsqueeze import Unsqueeze


class UnsqueezeInternal(Unsqueeze):
    @staticmethod
    def infer(node: Node):
        axis_value = node.in_port(1).data.get_value()
        Unsqueeze.infer(node)
        # preserve initial axis value
        node.in_port(1).data.set_value(axis_value)
