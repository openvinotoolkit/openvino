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

from extensions.front.sub import Sub
from mo.utils.unittest.graph import build_graph
from mo.utils.ir_engine.compare_graphs import compare_graphs

nodes_attributes = {
    'placeholder_1': {'shape': np.array([1, 227, 227, 3]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_2': {'shape': np.array([1, 227, 227, 3]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    # Sub operation
    'Sub': {'kind': 'op', 'op': 'Sub'},
    # Test operation
    'last': {'type': None, 'value': None, 'kind': 'op', 'op': None},
    # Add and Mul operations
    'const': {'kind': 'op', 'op': 'Const', 'value': np.array(-1)},
    'neg': {'type': 'Multiply', 'kind': 'op', 'op': 'Mul'},
    'add_1': {'value': None, 'type': 'Add', 'kind': 'op', 'op': 'Add'},
}


class TestSub(unittest.TestCase):
    def test_sub_test_1(self):
        # Test with two different inputs from two placeholders
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'Sub'),
                             ('placeholder_2', 'Sub'),
                             ('Sub', 'last')
                             ], nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_2', 'neg'),
                                 ('const', 'neg'),
                                 ('neg', 'add_1'),
                                 ('placeholder_1', 'add_1'),
                                 ('add_1', 'last'),
                                 ], nodes_with_edges_only=True)

        graph.stage = 'front'

        tested_class = Sub()
        tested_class.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'last', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_sub_test_2(self):
        # test with two same inputs from one placeholder
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'Sub'),
                             ('placeholder_1', 'Sub'),
                             ('Sub', 'last')
                             ], nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('neg', 'add_1'),
                                 ('placeholder_1', 'add_1'),
                                 ('placeholder_1', 'neg'),
                                 ('const', 'neg'),
                                 ('add_1', 'last'),
                                 ], nodes_with_edges_only=True)

        graph.stage = 'front'

        tested_class = Sub()
        tested_class.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'last', check_op_attrs=True)
        self.assertTrue(flag, resp)
