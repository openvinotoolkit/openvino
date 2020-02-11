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

from extensions.front.softsign_replacer import SoftSign
from mo.utils.unittest.graph import build_graph
from mo.utils.ir_engine.compare_graphs import compare_graphs

nodes_attributes = {
    'placeholder_1': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter', 'shape': np.array([1, 227, 227, 3])},
    # SquaredDifference operation
    'softsign': {'type': 'Softsign', 'kind': 'op', 'op': 'Softsign'},
    # Test operation
    'last': {'type': None, 'value': None, 'kind': 'op', 'op': None},
    # Mul and Add operations
    '1_const': {'kind': 'op', 'op': 'Const', 'value': np.array(1)},
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'add': {'kind': 'op', 'op': 'Add'},
    'div': {'type': 'Divide', 'kind': 'op', 'op': 'Div'},
}


class SoftsignTest(unittest.TestCase):
    def test_softsign_test(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'softsign'),
                             ('softsign', 'last'),
                             ], nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'abs', {'out': 0}),
                                 ('abs', 'add', {'in': 0}),
                                 ('1_const', 'add', {'in': 1}),
                                 ('placeholder_1', 'div', {'in': 0, 'out': 0}),
                                 ('add', 'div', {'in': 1}),
                                 ('div', 'last'),
                                 ], nodes_with_edges_only=True)

        graph.stage = 'front'

        replacer = SoftSign()
        replacer.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'last', check_op_attrs=True)
        self.assertTrue(flag, resp)
