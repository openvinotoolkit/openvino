# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.ops.elementwise import Mul
from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.const import Const


class SoftmaxFrontReplacementSubgraph(FrontReplacementSubgraph):
    enabled = True

    def pattern(self):
        return dict(
            nodes=[
                ('softmax', dict(type='SoftMax'))
            ],
            edges=[]
        )

    def replace_sub_graph(self, graph: Graph, match: dict):
        node = match['softmax']
        if 'temperature' in node and node['temperature'] != 1.0:
            in_node = node.in_node()
            out_nodes = [node for node in node.out_nodes().values()]
            graph.remove_edge(node.in_node().id, node.id)
            temperature = mo_array([1.0 / node.temperature])
            scalar_value_op = Const(graph, dict(value=temperature, shape=temperature.shape,
                                                symbol_dict={'name': node.id + '/const'}))
            mul_op = Mul(graph, dict(name=node.id + '/mul_', symbol_dict={'name': node.id + '/mul_'}))
            mul_node = mul_op.create_node(inputs=[in_node, scalar_value_op.create_node()])
            edge_attrs = graph.get_edge_data(node.id, out_nodes[0].id)[0]
            graph.add_edges_from([(mul_node.id, node.id, edge_attrs)])

