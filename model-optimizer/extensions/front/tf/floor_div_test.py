"""
 Copyright (C) 2018-2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import unittest

from extensions.front.tf.floor_div_decomposition import FloorDivDecomposition
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph, result, connect, \
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
