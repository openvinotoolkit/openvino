# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.front.softsign_replacer import SoftSign
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph

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
