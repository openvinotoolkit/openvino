# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.tf.UnpackPackReverseInputChannels import UnpackPackReverseInputChannels
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs

from unit_tests.utils.graph import build_graph, regular_op_with_empty_data, result, connect_front

nodes = {
    **regular_op_with_empty_data('input', {'type': 'Parameter'}),
    **regular_op_with_empty_data('unpack', {'op': 'AttributedSplit', 'axis': int64_array(0)}),
    **regular_op_with_empty_data('pack', {'op': 'Pack', 'axis': int64_array(0)}),
    **result(),

    **regular_op_with_empty_data('reverseChannels',
                                 {'op': 'ReverseChannels', 'order': int64_array([2, 1, 0]), 'axis': int64_array(0), 'type': None}),
}


class UnpackPackReverseInputChannelsTest(unittest.TestCase):
    def test_replace_to_reverse_channel(self):
        graph = build_graph(nodes_attrs=nodes, edges=[
            *connect_front('input:0', '0:unpack'),
            *connect_front('unpack:0', '2:pack'),
            *connect_front('unpack:1', '1:pack'),
            *connect_front('unpack:2', '0:pack'),
            *connect_front('pack:0', '0:output'),
        ], nodes_with_edges_only=True)
        graph.stage = 'front'

        UnpackPackReverseInputChannels().find_and_replace_pattern(graph)

        graph_ref = build_graph(nodes_attrs=nodes, edges=[
            *connect_front('input:0', '0:reverseChannels'),
            *connect_front('reverseChannels:0', '0:output'),
        ], nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)
