# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.middle.passes.eliminate import remove_op_node_with_data_node
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern


class ConstSwitchEraser(MiddleReplacementPattern):
    """
    Erases switch operation and its constant input after data and control flow infer
    """
    enabled = True

    def run_after(self):
        from openvino.tools.mo.middle.pass_separator import MiddleStart
        return [MiddleStart]

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.pseudo_topological_sort():
            if node.kind == 'data' or node.op != 'Switch':
                continue
            switch_op_node = node
            pred_id_data_node = switch_op_node.in_node(1)
            graph.remove_edge(pred_id_data_node.id, switch_op_node.id)
            remove_op_node_with_data_node(graph, switch_op_node)
