# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from extensions.back.ReverseInputChannels import ReverseChannelsPropagationUp, ReverseChannelsPropagationDown
from mo.front.common.partial_infer.utils import int64_array, float32_array
from mo.graph.graph import Node, Graph
from mo.middle.passes.eliminate import shape_inference
from mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, result, connect, regular_op_with_shaped_data, valued_const_with_data, \
    regular_op_with_empty_data

nodes = {
    **regular_op_with_shaped_data('placeholder1', [1, 3, 10, 10], {'type': 'Parameter'}),
    **regular_op_with_shaped_data('placeholder2', [1, 1, 1, 1], {'type': 'Parameter'}),

    **regular_op_with_shaped_data('mul', [1, 3, 10, 10], {'type': 'Multiply'}),
    **regular_op_with_shaped_data('reverse_channels', [1, 3, 10, 10], {'type': 'ReverseChannels', 'axis': 1}),


    **regular_op_with_shaped_data('pad', [1, 3, 10, 10], {'type': 'Pad'}),

    **result('result'),
}


nodes2 = {
    **regular_op_with_shaped_data('placeholder', [1, 3, 10, 10], {'type': 'Parameter'}),

    **valued_const_with_data('mul_const', float32_array([-127.5, -127.5, -127.5])),
    **regular_op_with_shaped_data('mul', [1, 3, 10, 10], {'type': 'Multiply'}),
    **valued_const_with_data('pad_const_1', int64_array([0, 0, 0, 0])),
    **valued_const_with_data('pad_const_2', int64_array([0, 0, 1, 1])),
    **regular_op_with_shaped_data('pad', [1, 3, 10, 10], {'type': 'Pad'}),
    **regular_op_with_shaped_data('reverse_channels', [1, 3, 10, 10], {'type': 'ReverseChannels', 'axis': 1}),
    **result('result'),
    **result('result2'),
}

nodes3 = {
    **regular_op_with_empty_data('placeholder', {'type': 'Parameter'}),
    **regular_op_with_empty_data('transpose', {'type': 'Transpose'}),
    **valued_const_with_data('transpose_order', int64_array([0, 3, 1, 2])),
    **regular_op_with_empty_data('reverse_channels_up', {'type': 'ReverseChannels', 'axis': 3}),
    **regular_op_with_empty_data('reverse_channels_down', {'type': 'ReverseChannels', 'axis': 1}),
    **result('result'),
    **result('result2'),
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

    def test_lift_up_through_pad(self):
        graph = build_graph(nodes2, [*connect('placeholder', '0:mul'), *connect('mul_const', '1:mul'),
                                     *connect('mul', '0:pad'), *connect('pad_const_1', '1:pad'),
                                     *connect('pad_const_2', '2:pad'), *connect('pad', 'reverse_channels'),
                                     *connect('reverse_channels', 'result')])
        self.set_graph_attrs(graph, ['placeholder'])

        node = Node(graph, 'pad')
        reverse_channels = Node(graph, 'reverse_channels')

        keep_moving_up, new_reverses = ReverseChannelsPropagationUp.lift_up_through(node, reverse_channels)
        self.assertTrue(keep_moving_up is True)
        self.assertTrue(len(new_reverses) == 1)
        self.check_graph_attrs(graph, ['placeholder'])

    def test_lift_up_through_pad2(self):
        graph = build_graph(nodes2, [*connect('placeholder', '0:mul'), *connect('mul_const', '1:mul'),
                                     *connect('mul', '0:pad'), *connect('pad_const_1', '1:pad'),
                                     *connect('pad_const_2', '2:pad'), *connect('pad', 'reverse_channels'),
                                     *connect('reverse_channels:0', '0:result'),  *connect('reverse_channels:0', '0:result2')])
        self.set_graph_attrs(graph, ['placeholder'])

        node = Node(graph, 'pad')
        reverse_channels = Node(graph, 'reverse_channels')

        keep_moving_up, new_reverses = ReverseChannelsPropagationUp.lift_up_through(node, reverse_channels)
        self.assertTrue(keep_moving_up is True)
        self.assertTrue(len(new_reverses) == 1)
        self.check_graph_attrs(graph, ['placeholder'])

    def test_pass_rc_through(self):
        graph = build_graph(nodes2, [*connect('placeholder', '0:mul'), *connect('mul_const', '1:mul'),
                                     *connect('mul', 'reverse_channels'),  *connect('reverse_channels', '0:pad'),
                                     *connect('pad_const_1', '1:pad'), *connect('pad_const_2', '2:pad'),
                                     *connect('pad', 'result')])
        self.set_graph_attrs(graph, ['placeholder'])

        node = Node(graph, 'pad')
        reverse_channels = Node(graph, 'reverse_channels')

        ReverseChannelsPropagationDown.pass_rc_through(node, reverse_channels)
        self.check_graph_attrs(graph, ['placeholder'])

    def test_lift_up_through_transpose(self):
        graph = build_graph(nodes3, [*connect('placeholder', '0:transpose'), *connect('transpose_order', '1:transpose'),
                                     *connect('transpose', 'reverse_channels_down'),
                                     *connect('reverse_channels_down', 'result')])
        graph_ref = build_graph(nodes3, [*connect('placeholder', 'reverse_channels_down'),
                                         *connect('transpose_order', '1:transpose'),
                                         *connect('reverse_channels_down', 'transpose'),
                                         *connect('transpose', 'result')])
        self.set_graph_attrs(graph, ['placeholder'])

        node = Node(graph, 'transpose')
        reverse_channels = Node(graph, 'reverse_channels_down')

        keep_moving_up, new_reverses = ReverseChannelsPropagationUp.lift_up_through_transpose(node, reverse_channels)
        self.assertTrue(keep_moving_up is True)
        self.assertTrue(len(new_reverses) == 1)
        self.check_graph_attrs(graph, ['placeholder'])
        (flag, resp) = compare_graphs(graph, graph_ref, 'result')
        self.assertTrue(flag, resp)

        reverse_channels = Node(graph, 'reverse_channels_down')
        self.assertTrue(reverse_channels.axis == 3)

    def test_lift_down_through_transpose(self):
        graph = build_graph(nodes3, [*connect('placeholder', 'reverse_channels_up'),
                                     *connect('transpose_order', '1:transpose'),
                                     *connect('reverse_channels_up', '0:transpose'),
                                     *connect('transpose', 'result')])
        graph_ref = build_graph(nodes3, [*connect('placeholder', '0:transpose'),
                                         *connect('transpose_order', '1:transpose'),
                                         *connect('transpose', 'reverse_channels_up'),
                                         *connect('reverse_channels_up', '0:result')])
        self.set_graph_attrs(graph, ['placeholder'])

        node = Node(graph, 'transpose')
        reverse_channels = Node(graph, 'reverse_channels_up')

        keep_moving_down = ReverseChannelsPropagationDown.pass_rc_through_transpose(node, reverse_channels)

        shape_inference(graph)

        self.assertTrue(keep_moving_down is True)
        self.check_graph_attrs(graph, ['placeholder'])
        (flag, resp) = compare_graphs(graph, graph_ref, 'result')
        self.assertTrue(flag, resp)

        reverse_channels = Node(graph, 'reverse_channels_down')
        self.assertTrue(reverse_channels.axis == 1)
