"""
 Copyright (c) 2018-2019 Intel Corporation

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

from extensions.front.div import Div
from mo.utils.unittest.graph import build_graph, compare_graphs

nodes_attributes = {
    'placeholder_1': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_2': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    # Div operation
    'Div': {'kind': 'op', 'op': 'Div'},
    # Test operation
    'last': {'type': None, 'value': None, 'kind': 'op', 'op': None},
    # Mul and Pow operations
    'const': {'value': np.array(-1), 'op': 'Const', 'kind': 'op'},
    'reciprocal': {'value': None, 'type': 'Pow', 'kind': 'op', 'op': 'Pow'},
    'mul_1': {'value': None, 'type': 'Multiply', 'kind': 'op', 'op': 'Mul'},
}


class TestDiv(unittest.TestCase):
    def test_div_test_1(self):
        # Test with two different inputs from two placeholders
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'Div'),
                             ('placeholder_2', 'Div'),
                             ('Div', 'last')
                             ],
                            {'placeholder_1': {'shape': np.array([1, 227, 227, 3])},
                             'placeholder_2': {'shape': np.array([1, 227, 227, 3])},
                             }, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_2', 'reciprocal', {'in': 0}),
                                 ('const', 'reciprocal', {'in': 1}),
                                 ('reciprocal', 'mul_1'),
                                 ('placeholder_1', 'mul_1'),
                                 ('mul_1', 'last'),
                                 ],
                                {'placeholder_1': {'shape': np.array([1, 227, 227, 3])},
                                 'placeholder_2': {'shape': np.array([1, 227, 227, 3])},
                                 }, nodes_with_edges_only=True)

        graph.stage = 'front'

        tested_class = Div()
        tested_class.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'last', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_div_test_2(self):
        # Test with two same inputs from one placeholder
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'Div'),
                             ('placeholder_1', 'Div'),
                             ('Div', 'last')
                             ],
                            {'placeholder_1': {'shape': np.array([1, 227, 227, 3])},
                             }, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('reciprocal', 'mul_1'),
                                 ('placeholder_1', 'mul_1'),
                                 ('placeholder_1', 'reciprocal', {'in': 0}),
                                 ('const', 'reciprocal', {'in': 1}),
                                 ('mul_1', 'last'),
                                 ],
                                {'placeholder_1': {'shape': np.array([1, 227, 227, 3])},
                                 }, nodes_with_edges_only=True)

        graph.stage = 'front'

        tested_class = Div()
        tested_class.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'last', check_op_attrs=True)
        self.assertTrue(flag, resp)
