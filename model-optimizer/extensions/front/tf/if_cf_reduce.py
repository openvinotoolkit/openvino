# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from extensions.ops.loop import Loop
from mo.front.common.replacement import FrontReplacementPattern
from mo.graph.graph import Graph, Node
from mo.ops.const import Const


class IfControlFlowReduce(FrontReplacementPattern):
    """
    Remove control flow edges from If
    """
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(op='If'):
            IfControlFlowReduce.delete_cf_edges(node)

    @staticmethod
    def delete_cf_edges(if_node):

        if len(if_node.out_ports()) == 0 or len(if_node.in_ports()) == 0:
            for in_port in if_node.in_ports(True).values():
                in_port.disconnect()

            for out_port in if_node.out_ports(True).values():
                out_port.disconnect()
            if_node.graph.erase_node(if_node)
            return

        # delete input control_flow ports
        for in_port in if_node.in_ports(True).values():
            if in_port.control_flow:
                in_port.disconnect()

        # delete input control_flow ports
        for out_port in if_node.out_ports(True).values():
            if out_port.control_flow:
                out_port.disconnect()
