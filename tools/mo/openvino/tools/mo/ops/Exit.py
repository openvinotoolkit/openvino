# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.op import Op


class Exit(Op):
    op = "Exit"

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': None,
            'op': __class__.op,
            'infer': Exit.exit_infer,
            'in_ports_count': 1,
        }
        super().__init__(graph, mandatory_props, attrs)

    @staticmethod
    def exit_infer(node: Node):
        output_shape = node.in_node(0).shape
        output_value = node.in_node(0).value
        for _, out_node in node.graph.out_edges(node.id):
            node.graph.node[out_node]['shape'] = mo_array(output_shape)
            node.graph.node[out_node]['value'] = None if output_value is None else mo_array(output_value)
