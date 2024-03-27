# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.ops.loop import Loop
from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph
from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.ops.const import Const


class WhileNormalize(FrontReplacementSubgraph):
    """
    Normalize inputs for Loop replacing TensorFlow 2 While operation:
    1) Remove external input port for current iteration
    2) Move trip count from port #1 to port #0
    3) Occupy port #1 for execution condition
    """
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(op='Loop'):
            self.normalize_loop_node(graph, node)

    @staticmethod
    def normalize_loop_node(graph: Graph, loop_node: Node):
        loop_name = loop_node.soft_get('name', loop_node.id)

        # disconnect current iteration from external port #0 and move trip count to this port
        loop_node.in_port(0).disconnect()
        loop_node.in_port(1).get_connection().add_destination(loop_node.in_port(0))
        Loop.update_port_map_value(loop_node.input_port_map, 'external_port_id', 1, 0)

        # connect execution condition port
        exec_cond_node = Const(graph, {'name': loop_name + '/ExecutionConditionValue',
                                       'value': mo_array(True, dtype=bool)}).create_node()
        loop_node.in_port(1).get_connection().set_source(exec_cond_node.out_port(0))

        loop_node.body.clean_up()
        Loop.normalize_input_output_ports(loop_node)
