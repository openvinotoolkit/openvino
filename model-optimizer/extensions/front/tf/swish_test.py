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

from extensions.front.tf.swish import Swish
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph

nodes_attributes = {
    'placeholder_1': {'shape': np.array([1, 227, 227, 3]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_2': {'shape': np.array([1, 227, 227, 3]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    # swish operation
    'swish': {'kind': 'op', 'op': 'swish_f32'},
    # Test operation
    'last': {'type': None, 'value': None, 'kind': 'op', 'op': None},
    # Add and Mul operations
    'mul': {'type': 'Multiply', 'kind': 'op', 'op': 'Mul'},
    'sigmoid': {'value': None, 'type': 'Sigmoid', 'kind': 'op', 'op': 'Sigmoid'},
}


class TestSwish(unittest.TestCase):
    def test_swish_test_1(self):
        # Test with two different inputs from two placeholders
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'swish'),
                             ('swish', 'last')
                             ], nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'sigmoid', {'out': 0}),
                                 ('placeholder_1', 'mul', {'in': 0, 'out': 0}),
                                 ('sigmoid', 'mul', {'in': 1}),
                                 ('mul', 'last'),
                                 ], nodes_with_edges_only=True)

        graph.stage = 'front'
        Swish().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'last', check_op_attrs=True)
        self.assertTrue(flag, resp)
