# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.ops.squeeze import Squeeze


class SqueezeInternal(Squeeze):
    @staticmethod
    def infer(node: Node):
        axis_value = node.in_port(1).data.get_value()
        Squeeze.infer(node)
        # preserve initial axis value
        node.in_port(1).data.set_value(axis_value)
