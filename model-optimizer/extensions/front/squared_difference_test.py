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

from extensions.front.squared_difference import SquaredDifference
from mo.utils.unittest.graph import build_graph
from mo.utils.ir_engine.compare_graphs import compare_graphs

nodes_attributes = {
    'placeholder_1': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter', 'shape': np.array([1, 227, 227, 3])},
    'placeholder_2': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter', 'shape': np.array([1, 227, 227, 3])},
    # SquaredDifference operation
    'squared_difference': {'type': 'SquaredDifference', 'kind': 'op', 'op': 'SquaredDifference'},
    # Test operation
    'last': {'type': None, 'value': None, 'kind': 'op', 'op': None},
    # Mul and Add operations
    'n_const': {'kind': 'op', 'op': 'Const', 'value': np.array(-1)},
    'negate_1': {'type': 'Multiply', 'kind': 'op', 'op': 'Mul'},
    's_const': {'kind': 'op', 'op': 'Const', 'value': np.array(2)},
    'square_1': {'type': 'Power', 'kind': 'op', 'op': 'Pow'},
    'add_1': {'value': None, 'operation': None, 'type': 'Add', 'kind': 'op'},
}


class SquaredDifferenceTest(unittest.TestCase):
    def test_squared_difference_test_1(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'squared_difference'),
                             ('placeholder_2', 'squared_difference'),
                             ('squared_difference', 'last'),
                             ], nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'add_1'),
                                 ('placeholder_2', 'negate_1'),
                                 ('n_const', 'negate_1'),
                                 ('negate_1', 'add_1'),
                                 ('add_1', 'square_1', {'in': 0}),
                                 ('s_const', 'square_1', {'in': 1}),
                                 ('square_1', 'last'),
                                 ], nodes_with_edges_only=True)

        graph.stage = 'front'

        replacer = SquaredDifference()
        replacer.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'last', check_op_attrs=True)
        self.assertTrue(flag, resp)
