# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.front.Log1p import Log1p
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph

nodes_attributes = {
    'placeholder': {'shape': np.array([4, 5, 6]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    # Log1p operation
    'Log1p': {'kind': 'op', 'op': 'Log1p'},
    # Test operation
    'last': {'type': None, 'value': None, 'kind': 'op', 'op': None},
    # Add and Log operations
    'const': {'kind': 'op', 'op': 'Const', 'value': np.ones([1], dtype=np.float32)},
    'add': {'type': 'Add', 'kind': 'op', 'op': 'Add'},
    'log': {'type': 'Log', 'kind': 'op', 'op': 'Log'},
}


class TestLog1p(unittest.TestCase):
    def test_log1p_test(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder', 'Log1p'),
                             ('Log1p', 'last')
                             ], nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('const', 'add'),
                                 ('placeholder', 'add'),
                                 ('add', 'log'),
                                 ('log', 'last'),
                                 ], nodes_with_edges_only=True)

        graph.stage = 'front'

        tested_class = Log1p()
        tested_class.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'last', check_op_attrs=True)
        self.assertTrue(flag, resp)
