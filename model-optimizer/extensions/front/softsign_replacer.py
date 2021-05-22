# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from extensions.ops.activation_ops import Abs
from extensions.ops.elementwise import Add, Div
from mo.front.common.replacement import FrontReplacementOp
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Graph, Node


class SoftSign(FrontReplacementOp):
    enabled = True
    op = "Softsign"

    def replace_op(self, graph: Graph, node: Node):
        """
        Replace Softsign according to formula feature/(abs(feature)+1)
        """
        abs_node = Abs(graph, {'name': "abs_" + node.id}).create_node()
        abs_node.in_port(0).connect(node.in_port(0).get_source())

        add_node = create_op_node_with_second_input(graph, Add, np.ones([1]), {"name": node.id + "_plus_1"})
        add_node.in_port(0).connect(abs_node.out_port(0))
        div_node = Div(graph, {"name": "div_" + node.id}).create_node()
        div_node.in_port(0).connect(node.in_port(0).get_source())
        div_node.in_port(1).connect(add_node.out_port(0))
        return [div_node.id]
