# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.utils import shape_array
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.op import Op


class Enter(Op):
    op = "Enter"

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': None,
            'op': self.op,
            'in_ports_count': 1,
            'infer': Enter.enter_infer,
        }
        super().__init__(graph, mandatory_props, attrs)

    @staticmethod
    def enter_infer(node: Node):
        output_shape = node.in_port(0).data.get_shape()
        output_value = node.in_port(0).data.get_value()

        for _, out_node in node.graph.out_edges(node.id):
            node.graph.node[out_node]['shape'] = shape_array(output_shape)
            node.graph.node[out_node]['value'] = None if output_value is None else output_value.copy()
