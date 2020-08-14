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

from generator import generator, generate

from extensions.back.ReduceL1L2ToReduceLp import ReduceL1L2ToReduceLp
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph, result, regular_op_with_shaped_data, regular_op_with_empty_data, \
    valued_const_with_data, connect


@generator
class TestReduceL1L2ToReduceLp(unittest.TestCase):
    @generate(*[('ReduceL1', 1), ('ReduceL2', 2)])
    def test_reduce_l1_to_reduce_lp(self, op: str, p: int):
        nodes = {
            **regular_op_with_shaped_data('input', [1, 3, 224, 224], {'type': 'Parameter'}),

            **valued_const_with_data('axes', int64_array([2, 3])),
            **valued_const_with_data('p', int64_array(p)),

            **regular_op_with_empty_data('reduce', {'op': op, 'type': None, 'name': 'reduce_name'}),
            **regular_op_with_empty_data('reduce_lp', {'op': 'ReduceLp', 'type': 'ReduceLp', 'name': 'reduce_name'}),
            **result(),
        }
        graph = build_graph(nodes, [
            *connect('input', '0:reduce'),
            *connect('axes', '1:reduce'),
            *connect('reduce', 'output'),
        ], nodes_with_edges_only=True)

        ReduceL1L2ToReduceLp().find_and_replace_pattern(graph)

        graph_ref = build_graph(nodes, [
            *connect('input', '0:reduce_lp'),
            *connect('axes', '1:reduce_lp'),
            *connect('p', '2:reduce_lp'),
            *connect('reduce_lp', 'output'),
        ], nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, last_node='output', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.assertTrue(Node(graph_ref, 'reduce_lp').name == 'reduce_name')
