# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.front.tf.floor_div_decomposition import FloorDivDecomposition
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, result, connect, \
    connect_data, regular_op_with_empty_data

nodes = {
    **regular_op_with_empty_data('placeholder_1', {'type': 'Parameter', 'op': 'Parameter'}),
    **regular_op_with_empty_data('placeholder_2', {'type': 'Parameter', 'op': 'Parameter'}),
    **regular_op_with_empty_data('floor_div', {'op': 'FloorDiv', 'type': None, 'name': 'my_floor_div'}),

    **regular_op_with_empty_data('div', {'type': 'Divide', 'op': 'Div'}),
    **regular_op_with_empty_data('floor', {'type': 'Floor', 'op': 'Floor'}),

    **result(),
}


class TestFloorDiv(unittest.TestCase):
    def test_floor_div_test_1(self):
        # Test with two different inputs from two placeholders
        graph = build_graph(nodes, [
            *connect('placeholder_1:0', '0:floor_div'),
            *connect('placeholder_2:0', '1:floor_div'),
            *connect('floor_div:0', '0:output'),
        ], nodes_with_edges_only=True)
        FloorDivDecomposition().find_and_replace_pattern(graph)

        graph_ref = build_graph(nodes, [
            *connect('placeholder_1:0', '0:div'),
            *connect('placeholder_2:0', '1:div'),
            *connect('div:0', '0:floor'),
            *connect('floor:0', '0:output'),
        ], nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.assertTrue(graph.node[graph.get_nodes_with_attributes(type='Floor')[0]]['name'] == 'my_floor_div')

    def test_floor_div_test_2(self):
        # Test with two same inputs from one placeholder
        graph = build_graph(nodes, [
            *connect('placeholder_1:0', '0:floor_div'),
            *connect_data('placeholder_1:0', '1:floor_div'),
            *connect('floor_div', 'output'),
        ], nodes_with_edges_only=True)
        FloorDivDecomposition().find_and_replace_pattern(graph)

        graph_ref = build_graph(nodes, [
            *connect('placeholder_1:0', '0:div'),
            *connect_data('placeholder_1:0', '1:div'),
            *connect('div', 'floor'),
            *connect('floor', 'output'),
        ], nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.assertTrue(graph.node[graph.get_nodes_with_attributes(type='Floor')[0]]['name'] == 'my_floor_div')
