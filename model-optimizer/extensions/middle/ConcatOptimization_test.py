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

from extensions.middle.ConcatOptimization import ConcatOdInputEraser
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph, result, regular_op_with_shaped_data, shaped_const_with_data, connect


class Concat0dInput(unittest.TestCase):
    def test_deletion(self):
        nodes = {
            **shaped_const_with_data('input_0', [1]),
            **shaped_const_with_data('input_1', [1]),
            **shaped_const_with_data('input_2', [0]),
            **shaped_const_with_data('input_3', [1]),
            **regular_op_with_shaped_data('concat', [3], {'type': 'Concat'}),
            **result(),
        }
        edges_before = [
            *connect('input_0', '0:concat'),
            *connect('input_1', '1:concat'),
            *connect('input_2', '2:concat'),
            *connect('input_3', '3:concat'),
            *connect('concat', 'output'),
        ]
        edges_after = [
            *connect('input_0', '0:concat'),
            *connect('input_1', '1:concat'),
            *connect('input_3', '3:concat'),
            *connect('concat', 'output'),
        ]

        graph = build_graph(nodes, edges_before, nodes_with_edges_only=True)
        ConcatOdInputEraser().find_and_replace_pattern(graph)
        graph_ref = build_graph(nodes, edges_after, nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_negative(self):
        nodes = {
            **shaped_const_with_data('input_0', [1]),
            **shaped_const_with_data('input_1', [1]),
            **shaped_const_with_data('input_2', [1]),
            **shaped_const_with_data('input_3', [1]),
            **regular_op_with_shaped_data('concat', [4], {'type': 'Concat'}),
            **result(),
        }
        edges = [
            *connect('input_0', '0:concat'),
            *connect('input_1', '1:concat'),
            *connect('input_2', '2:concat'),
            *connect('input_3', '3:concat'),
            *connect('concat', 'output'),
        ]

        graph = build_graph(nodes, edges, nodes_with_edges_only=True)
        ConcatOdInputEraser().find_and_replace_pattern(graph)
        graph_ref = build_graph(nodes, edges, nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_assertion_error(self):
        nodes = {
            **shaped_const_with_data('input_0', [0]),
            **shaped_const_with_data('input_1', [0]),
            **shaped_const_with_data('input_2', [0]),
            **shaped_const_with_data('input_3', [0]),
            **regular_op_with_shaped_data('concat', [0], {'type': 'Concat'}),
            **result(),
        }
        edges = [
            *connect('input_0', '0:concat'),
            *connect('input_1', '1:concat'),
            *connect('input_2', '2:concat'),
            *connect('input_3', '3:concat'),
            *connect('concat', 'output'),
        ]

        graph = build_graph(nodes, edges, nodes_with_edges_only=True)
        self.assertRaises(AssertionError, ConcatOdInputEraser().find_and_replace_pattern, graph)
