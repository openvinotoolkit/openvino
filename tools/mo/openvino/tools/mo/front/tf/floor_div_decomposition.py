# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.activation_ops import Floor
from openvino.tools.mo.ops.elementwise import Div
from openvino.tools.mo.front.common.replacement import FrontReplacementPattern
from openvino.tools.mo.graph.graph import Graph, Node, rename_node


class FloorDivDecomposition(FrontReplacementPattern):
    r"""
    BEFORE:                     AFTER:
    input_0     input_1         input_0     input_1
        \       /                   \       /
        FloorDiv                       Div
            |                           |
          output                      Floor
                                        |
                                      output
    """
    enabled = True

    @staticmethod
    def floor_div_replacement(floor_div: Node):
        graph = floor_div.graph
        name = floor_div.soft_get('name', floor_div.id)

        div = Div(graph, {'name': name + '/Div'}).create_node()
        floor = Floor(graph, {'name': name}).create_node()
        div.out_port(0).connect(floor.in_port(0))

        div.in_port(0).connect(floor_div.in_port(0).get_source())
        div.in_port(1).connect(floor_div.in_port(1).get_source())
        floor_div.out_port(0).get_connection().set_source(floor.out_port(0))

        graph.remove_node(floor_div.id)
        rename_node(floor, name)

    def find_and_replace_pattern(self, graph: Graph):
        for floor_div in graph.get_op_nodes(op='FloorDiv'):
            self.floor_div_replacement(floor_div)
