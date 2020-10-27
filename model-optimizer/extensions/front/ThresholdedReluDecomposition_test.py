"""
 Copyright (C) 2020 Intel Corporation

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

from extensions.front.ThresholdedReluDecomposition import ThresholdedReluDecomposition
from mo.front.common.partial_infer.utils import float_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph, const

nodes_attributes = {
    'parameter': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'trelu': {'type': None, 'kind': 'op', 'op': 'ThresholdedRelu', 'alpha': 0.75, 'name': 'my_trelu'},
    'result': {'type': 'Result', 'value': None, 'kind': 'op', 'op': 'Result'},

    'cast': {'type': 'Convert', 'kind': 'op', 'op': 'Cast'},
    'greater': {'type': 'Greater', 'kind': 'op', 'op': 'Greater'},
    'mul': {'type': 'Multiply', 'kind': 'op', 'op': 'Mul', 'name': 'my_trelu'},
    'squeeze2': {'type': 'Squeeze', 'kind': 'op', 'op': 'Squeeze'},
    **const('alpha', float_array([0.75])),
}


class ThresholdedReluDecompositionTest(unittest.TestCase):
    def test_trelu(self):
        graph = build_graph(nodes_attributes,
                            [('parameter', 'trelu', {'in': 0, 'out': 0}),
                             ('trelu', 'result', {'in': 0, 'out': 0}),
                             ], nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('parameter', 'greater', {'in': 0, 'out': 0}),
                                 ('alpha', 'greater', {'in': 1, 'out': 0}),
                                 ('greater', 'cast', {'in': 0, 'out': 0}),
                                 ('parameter', 'mul', {'in': 0, 'out': 0}),
                                 ('cast', 'mul', {'in': 1, 'out': 0}),
                                 ('mul', 'result', {'in': 0, 'out': 0}),
                                 ], nodes_with_edges_only=True)

        graph.graph['layout'] = 'NCHW'
        graph.stage = 'front'

        ThresholdedReluDecomposition().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.assertTrue(len(graph.get_op_nodes(name='my_trelu')) == 1 and
                        graph.get_op_nodes(name='my_trelu')[0].op == 'Mul')
