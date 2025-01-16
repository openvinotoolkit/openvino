# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy

from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern


class ConvertMultiInputConv(MiddleReplacementPattern):
    enabled = True
    force_clean_up = True

    def run_after(self):
        from openvino.tools.mo.middle.pass_separator import PreMiddleStart
        return [PreMiddleStart]

    def run_before(self):
        from openvino.tools.mo.middle.pass_separator import MiddleStart
        return [MiddleStart]

    def pattern(self):
        return dict(
            nodes=[('op', dict(kind='op', op='ConvND'))],
            edges=[]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        node = match['op']
        node.op = 'Conv2D'

        if node.bias_term:
            num_inputs = len(node.in_nodes()) - 2
            w_node = node.in_node(len(node.in_nodes()) - 2)
            b_node = node.in_node(len(node.in_nodes()) - 1)
        else:
            num_inputs = len(node.in_nodes()) - 1
            w_node = node.in_node(len(node.in_nodes()) - 1)

        for i in range(1, num_inputs):
            in_i = node.in_node(i)
            out_i = node.out_node(i)
            conv_id = graph.unique_id(node.id + '__')
            graph.add_node(conv_id, **copy.deepcopy(node.get_attrs()))
            new_conv = Node(graph, conv_id)
            new_conv.name = conv_id

            graph.remove_edge(in_i.id, node.id)
            graph.remove_edge(node.id, out_i.id)
            graph.add_edges_from([
                (w_node.id, conv_id, {'in': 1, 'bin': 'weights'}),
            ])

            if node.bias_term:
                graph.add_edges_from([
                    (b_node.id, conv_id, {'in': 2, 'bin': 'biases'}),
                ])

            graph.add_edges_from([
                (in_i.id, conv_id, {'in': 0}),
            ])
            graph.add_edge(conv_id, out_i.id, **{'out': 0})
