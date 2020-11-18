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

import numpy as np

from extensions.front.reduce_axis_normalizer import ReduceAxisNormalizer
from mo.front.common.partial_infer.utils import int64_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph, result, connect_front, regular_op

nodes = {
    **regular_op('parameter', {'type': 'Parameter'}),
    **regular_op('reduce', {'op': 'ReduceSum', 'axis': None}),
    **regular_op('axis', {'op': 'Const', 'type': 'Const', 'value': int64_array([1])}),
    **result(),
}

edges = [
    *connect_front('parameter:0', '0:reduce'),
    *connect_front('reduce', 'output'),
]


class ReduceAxisNormalizerTest(unittest.TestCase):
    def test_reduce_axis_is_None(self):
        graph = build_graph(nodes, edges, nodes_with_edges_only=True)
        graph.stage = 'front'

        ReduceAxisNormalizer().find_and_replace_pattern(graph)

        ref_nodes = nodes.copy()
        ref_nodes.update({**regular_op('rank', {'op': 'Rank', 'type': None}),
                          **regular_op('range', {'op': 'Range', 'type': 'Range'}),
                          **regular_op('begin', {'type': 'Const', 'value': int64_array([0])}),
                          **regular_op('step', {'type': 'Const', 'value': int64_array([1])}),
                          })
        graph_ref = build_graph(ref_nodes, [
            *edges,
            *connect_front('parameter:0', 'rank'),
            *connect_front('begin:0', '0:range'),
            *connect_front('rank:0', '1:range'),
            *connect_front('step:0', '2:range'),
            *connect_front('range:0', '1:reduce'),
        ], nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_reduce_axis_is_const(self):
        graph = build_graph(nodes, edges, {'reduce': {'axis': 1}}, nodes_with_edges_only=True)
        graph.stage = 'front'

        graph_ref = build_graph(nodes, [
            *edges,
            *connect_front('axis', '1:reduce'),
        ], {'axis': {'value': np.int64(1)}}, nodes_with_edges_only=True)

        ReduceAxisNormalizer().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)
