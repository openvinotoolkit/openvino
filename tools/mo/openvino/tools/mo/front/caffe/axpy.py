# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.elementwise import Add
from openvino.tools.mo.front.common.replacement import FrontReplacementOp
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.scale_shift import ScaleShiftOp


class AxpyToSSandAdd(FrontReplacementOp):
    """
    Replaces Axpy layer with ScaleShift and Add.
    """
    op = "Axpy"
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        in_node_0 = node.in_node(0)
        in_node_1 = node.in_node(1)
        in_node_2 = node.in_node(2)

        ss = ScaleShiftOp(graph, {'name': node.id + "/ScaleShift_", 'axis': 0})
        scale_shift = ss.create_node(inputs=[in_node_1, in_node_0])

        el = Add(graph, {'name': node.id + "/Add_"})
        el_node = el.create_node(inputs=[scale_shift, in_node_2])

        return [el_node.id]
