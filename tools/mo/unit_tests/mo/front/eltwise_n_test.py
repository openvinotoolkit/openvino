# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.front.eltwise_n import EltwiseNReplacement
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph

nodes_attributes = {
    'placeholder_1': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_2': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_3': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_4': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    # EltwiseN operations
    'EltwiseN_1': {'value': None, 'operation': None, 'type': None, 'kind': 'op', 'op': 'EltwiseN'},
    # Test operation
    'last': {'value': None, 'operation': None, 'type': 'Eltwise', 'kind': 'op', 'op': None},
    # Max operation
    'max_1': {'value': None, 'operation': None, 'type': 'Eltwise', 'kind': 'op'},
    # Mui operations
    'mul_1': {'value': None, 'operation': None, 'type': 'Eltwise', 'kind': 'op'},
    'mul_2': {'value': None, 'operation': None, 'type': 'Eltwise', 'kind': 'op'},
    # Add operations
    'add_1': {'value': None, 'operation': None, 'type': 'Eltwise', 'kind': 'op'},
    'add_2': {'value': None, 'operation': None, 'type': 'Eltwise', 'kind': 'op'},
    'add_3': {'value': None, 'operation': None, 'type': 'Eltwise', 'kind': 'op'},
}


class TestEltwiseNFrontReplacement(unittest.TestCase):
    def test_eltwiseN_test_1(self):
        # EltwiseN test with N = 2 from 2 placeholders and operation = max

        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'EltwiseN_1'),
                             ('placeholder_2', 'EltwiseN_1'),
                             ('EltwiseN_1', 'last')
                             ],
                            {'placeholder_1': {'shape': np.array([1, 227, 227, 3])},
                             'placeholder_2': {'shape': np.array([1, 227, 227, 3])},
                             'EltwiseN_1': {'operation': 'max'},
                             }, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'max_1'),
                                 ('placeholder_2', 'max_1'),
                                 ('max_1', 'last')
                                 ],
                                {'placeholder_1': {'shape': np.array([1, 227, 227, 3])},
                                 'placeholder_2': {'shape': np.array([1, 227, 227, 3])},
                                 'max_1': {'type': 'Maximum'}
                                 }, nodes_with_edges_only=True)

        graph.stage = 'front'

        replacer = EltwiseNReplacement()
        replacer.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'last', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_eltwise_test_2(self):
        # EltwiseN test with N = 3 from 3 placeholders and operation = mul

        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'EltwiseN_1'),
                             ('placeholder_2', 'EltwiseN_1'),
                             ('placeholder_3', 'EltwiseN_1'),
                             ('EltwiseN_1', 'last')
                             ],
                            {'placeholder_1': {'shape': np.array([1, 227, 227, 3])},
                             'placeholder_2': {'shape': np.array([1, 227, 227, 3])},
                             'placeholder_3': {'shape': np.array([1, 227, 227, 3])},
                             'EltwiseN_1': {'operation': 'mul'},
                             }, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'mul_1'),
                                 ('placeholder_2', 'mul_1'),
                                 ('mul_1', 'mul_2'),
                                 ('placeholder_3', 'mul_2'),
                                 ('mul_2', 'last')
                                 ],
                                {'placeholder_1': {'shape': np.array([1, 227, 227, 3])},
                                 'placeholder_2': {'shape': np.array([1, 227, 227, 3])},
                                 'placeholder_3': {'shape': np.array([1, 227, 227, 3])},
                                 'mul_1': {'type': 'Multiply'},
                                 'mul_2': {'type': 'Multiply'},
                                 }, nodes_with_edges_only=True)

        graph.stage = 'front'

        replacer = EltwiseNReplacement()
        replacer.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'last', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_eltwise_test_3(self):
        # EltwiseN test with N = 4 from 1 placeholder and operation = sum
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'EltwiseN_1'),
                             ('placeholder_2', 'EltwiseN_1'),
                             ('placeholder_3', 'EltwiseN_1'),
                             ('placeholder_4', 'EltwiseN_1'),
                             ('EltwiseN_1', 'last')
                             ],
                            {'placeholder_1': {'shape': np.array([1, 227, 227, 3])},
                             'placeholder_2': {'shape': np.array([1, 227, 227, 3])},
                             'placeholder_3': {'shape': np.array([1, 227, 227, 3])},
                             'placeholder_4': {'shape': np.array([1, 227, 227, 3])},
                             'EltwiseN_1': {'operation': 'sum'},
                             }, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'add_1'),
                                 ('placeholder_2', 'add_1'),
                                 ('add_1', 'add_2'),
                                 ('placeholder_3', 'add_2'),
                                 ('add_2', 'add_3'),
                                 ('placeholder_4', 'add_3'),
                                 ('add_3', 'last')
                                 ],
                                {'placeholder_1': {'shape': np.array([1, 227, 227, 3])},
                                 'placeholder_2': {'shape': np.array([1, 227, 227, 3])},
                                 'placeholder_3': {'shape': np.array([1, 227, 227, 3])},
                                 'placeholder_4': {'shape': np.array([1, 227, 227, 3])},
                                 'add_1': {'type': 'Add'},
                                 'add_2': {'type': 'Add'},
                                 'add_3': {'type': 'Add'},
                                 }, nodes_with_edges_only=True)

        graph.stage = 'front'

        replacer = EltwiseNReplacement()
        replacer.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'last', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_eltwise_test_4(self):
        # EltwiseN test with N = 4 from 1 placeholder and operation = sum
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'EltwiseN_1'),
                             ('placeholder_1', 'EltwiseN_1'),
                             ('placeholder_1', 'EltwiseN_1'),
                             ('placeholder_1', 'EltwiseN_1'),
                             ('EltwiseN_1', 'last')
                             ],
                            {'placeholder_1': {'shape': np.array([1, 227, 227, 3])},
                             'EltwiseN_1': {'operation': 'sum'},
                             }, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'add_1'),
                                 ('placeholder_1', 'add_1'),
                                 ('add_1', 'add_2'),
                                 ('placeholder_1', 'add_2'),
                                 ('add_2', 'add_3'),
                                 ('placeholder_1', 'add_3'),
                                 ('add_3', 'last')
                                 ],
                                {'placeholder_1': {'shape': np.array([1, 227, 227, 3])},
                                 'add_1': {'type': 'Add'},
                                 'add_2': {'type': 'Add'},
                                 'add_3': {'type': 'Add'},
                                 }, nodes_with_edges_only=True)

        graph.stage = 'front'

        replacer = EltwiseNReplacement()
        replacer.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'last', check_op_attrs=True)
        self.assertTrue(flag, resp)
