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

from extensions.front.reciprocal import ReciprocalReplacer
from mo.utils.unittest.graph import build_graph
from mo.utils.ir_engine.compare_graphs import compare_graphs

nodes_attributes = {
    'placeholder_1': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    # Reciprocal operation
    'reciprocal_1': {'kind': 'op', 'op': 'Reciprocal'},
    # Test operation
    'last': {'type': None, 'value': None, 'kind': 'op', 'op': None},
    # Pow operations
    'const': {'value': np.array(-1), 'op': 'Const', 'kind': 'op'},
    'pow': {'type': 'Power', 'kind': 'op', 'op': 'Pow'},
}


class ReciprocalReplacerTests(unittest.TestCase):
    def test_reciprocal_test_1(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'reciprocal_1'),
                             ('reciprocal_1', 'last')
                             ],
                            {'placeholder_1': {'shape': np.array([1, 227, 227, 3])},
                             }, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'pow', {'in': 0}),
                                 ('const', 'pow', {'in': 1}),
                                 ('pow', 'last'),
                                 ],
                                {'placeholder_1': {'shape': np.array([1, 227, 227, 3])},
                                 }, nodes_with_edges_only=True)

        graph.stage = 'front'

        pattern = ReciprocalReplacer()
        pattern.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'last', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_neg_reciprocal_1(self):
        # Test if power = 0

        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'reciprocal_1'),
                             ('reciprocal_1', 'last')
                             ],
                            {'placeholder_1': {'shape': np.array([1, 227, 227, 3])},
                             }, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'pow'),
                                 ('const', 'pow', {'in': 1}),
                                 ('pow', 'last'),
                                 ],
                                {'placeholder_1': {'shape': np.array([1, 227, 227, 3])},
                                 'const': {'value': np.array(0)},
                                 }, nodes_with_edges_only=True)

        graph.stage = 'front'

        pattern = ReciprocalReplacer()
        pattern.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'last', check_op_attrs=True)
        self.assertTrue(not flag)
