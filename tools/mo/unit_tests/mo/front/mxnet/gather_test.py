# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.front.mxnet.gather import GatherFrontReplacer
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph


class GatherTest(unittest.TestCase):
    def test_embedding_replace1(self):
        graph = build_graph(
            {'placeholder_1': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
             'embedding_const': {'value': None, 'shape': None, 'kind': 'op', 'op': 'Const'},
             'embedding': {'type': None, 'kind': 'op', 'op': 'Embedding'},
             'last': {'type': None, 'kind': 'op', 'op': None},
             },
            [
                ('placeholder_1', 'embedding', {'out': 0, 'in': 0}),
                ('embedding_const', 'embedding', {'out': 0, 'in': 1}),
                ('embedding', 'last')
            ],
            {
                'placeholder_1': {'shape': np.array([32, 35])},
                'embedding_const': {'shape': np.array([2000, 650]),
                                    'bias': np.array(np.random.randint(0, 225, (2000, 650)))},
            }, nodes_with_edges_only=True)

        graph_ref = build_graph(
            {'placeholder_1': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
             'embedding_const': {'value': None, 'kind': 'op', 'op': 'Const'},
             'axis_const': {'value': 0, 'kind': 'op', 'data_type': None,
                            'type': 'Const', 'op': 'Const'},
             'embedding': {'kind': 'op', 'op': 'Gather'},
             'last': {'type': None, 'kind': 'op', 'op': None},
             },
            [
                ('embedding_const', 'embedding', {'in': 1}),
                ('axis_const', 'embedding', {'in': 2}),
                ('placeholder_1', 'embedding', {'in': 0}),
                ('embedding', 'last')
            ],
            {'placeholder_1': {'shape': np.array([32, 35])},
             'embedding_const': {'shape': np.array([2000, 650]),
                                 'bias': np.array(np.random.randint(0, 225, (2000, 650)))},
             }, nodes_with_edges_only=True)

        graph.graph['layout'] = 'NCHW'
        graph.stage = 'front'

        replacer = GatherFrontReplacer()
        replacer.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'last')
        self.assertTrue(flag, resp)
