# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.front.reciprocal import ReciprocalReplacer
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph

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
