# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from extensions.back.ReverseInputChannels import ReverseChannelsPropagationUp
from mo.graph.graph import Node, Graph
from unit_tests.utils.graph import build_graph, result, connect, regular_op_with_shaped_data

nodes = {
    **regular_op_with_shaped_data('placeholder1', [1, 3, 10, 10], {'type': 'Parameter'}),
    **regular_op_with_shaped_data('placeholder2', [1, 1, 1, 1], {'type': 'Parameter'}),

    **regular_op_with_shaped_data('mul', [1, 3, 10, 10], {'type': 'Multiply'}),
    **regular_op_with_shaped_data('reverse_channels', [1, 3, 10, 10], {'type': 'ReverseChannels', 'axis': 1}),

    **result('result'),
}


class ReverseInputChannelsTest(unittest.TestCase):
    def check_graph_attrs(self, graph: Graph, parameter_node_names: list):
        for node in graph.get_op_nodes():
            if node.soft_get('name') in parameter_node_names:
                self.assertTrue(node.soft_get('type') == 'Parameter')
                out_node = node.out_node(0)
                self.assertTrue(out_node['fw_tensor_debug_info'] == ['fw_name', 0])
            else:
                for idx in node.out_nodes():
                    out_node = node.out_node(idx)
                    self.assertFalse('fw_tensor_debug_info' in out_node)

    def set_graph_attrs(self, graph: Graph, parameter_node_names: list):
        for node in graph.get_op_nodes():
            if node.soft_get('name') in parameter_node_names:
                self.assertTrue(node.soft_get('type') == 'Parameter')
                out_node = node.out_node(0)
                out_node['fw_tensor_debug_info'] = ['fw_name', 0]

    def test_lift_up_through_eltwise(self):
        graph = build_graph(nodes, [*connect('placeholder1', '0:mul'), *connect('placeholder2', '1:mul'),
                                    *connect('mul', 'reverse_channels'), *connect('reverse_channels', 'result')])
        self.set_graph_attrs(graph, ['placeholder1', 'placeholder2'])

        node = Node(graph, 'mul')
        reverse_channels = Node(graph, 'reverse_channels')

        ReverseChannelsPropagationUp.lift_up_through_eltwise(node, reverse_channels)
        self.check_graph_attrs(graph, ['placeholder1', 'placeholder2'])
