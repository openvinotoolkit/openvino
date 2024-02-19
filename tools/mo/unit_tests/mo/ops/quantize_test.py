# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.ops.fakequantize import FakeQuantize, broadcastable
from openvino.tools.mo.graph.graph import Node
from unit_tests.utils.graph import build_graph


class TestBroadcastable(unittest.TestCase):
    def test_matching(self):
        self.assertTrue(broadcastable([1, 2, 3], [1, 2, 3]))

    def test_incomplete(self):
        self.assertTrue(broadcastable([1, 1, 1], [1, 2, 3]))
        self.assertTrue(broadcastable([2, 3], [1, 2, 3]))
        self.assertTrue(broadcastable([1, 3], [1, 2, 3]))
        self.assertTrue(broadcastable([1, 1], [1, 2, 3]))
        self.assertTrue(broadcastable([], [1, 2, 3]))
        self.assertTrue(broadcastable([1], [1, 2, 3]))

    def test_reverse_incomplete(self):
        self.assertFalse(broadcastable([1, 2, 3], [1, 1, 1]))
        self.assertFalse(broadcastable([1, 2, 3], [2, 3]))
        self.assertFalse(broadcastable([1, 2, 3], [1, 3]))
        self.assertFalse(broadcastable([1, 2, 3], [1, 1]))
        self.assertFalse(broadcastable([1, 2, 3], []))
        self.assertFalse(broadcastable([1, 2, 3], [1]))

    def test_invalid(self):
        self.assertFalse(broadcastable([3, 2, 1], [1, 2, 3]))
        self.assertFalse(broadcastable([5], [6]))
        self.assertFalse(broadcastable([5], [1]))
        self.assertFalse(broadcastable([64], [1, 55, 56, 56]))


nodes_attributes = {'node_in_1': {'op': 'Identity', 'kind': 'op'},
                    'node_in_2': {'op': 'Identity', 'kind': 'op'},
                    'node_in_3': {'op': 'Identity', 'kind': 'op'},
                    'node_in_4': {'op': 'Identity', 'kind': 'op'},
                    'node_in_5': {'op': 'Identity', 'kind': 'op'},
                    'quantize': {'op': 'FakeQuantize', 'kind': 'op', 'levels': 2},
                    'node_out_1': {'op': 'Identity', 'kind': 'op'},
                    'op_output': {'kind': 'op', 'op': 'Result'}
                    }


class TestFakeQuantizeOp(unittest.TestCase):
    def test_shape_only(self):
        graph = build_graph(nodes_attributes,
                            [('node_in_1', 'quantize'),
                             ('node_in_2', 'quantize'),
                             ('node_in_3', 'quantize'),
                             ('node_in_4', 'quantize'),
                             ('node_in_5', 'quantize'),
                             ('quantize', 'node_out_1'),
                             ('node_out_1', 'op_output')
                             ],
                            {'node_out_1': {'shape': None},
                             'node_in_1': {'shape': np.array([1, 3, 10, 20])},
                             'node_in_2': {'shape': np.array([1, 3, 10, 20])},
                             'node_in_3': {'shape': np.array([1, 3, 10, 20])},
                             'node_in_4': {'shape': np.array([1, 3, 10, 20])},
                             'node_in_5': {'shape': np.array([1, 3, 10, 20])},
                             })

        quantize_node = Node(graph, 'quantize')
        FakeQuantize.infer(quantize_node)
        quantize_shape = np.array([1, 3, 10, 20])
        res_shape = graph.node['node_out_1']['shape']
        for i in range(0, len(quantize_shape)):
            self.assertEqual(quantize_shape[i], res_shape[i])

    def test_shape_and_value(self):
        graph = build_graph(nodes_attributes,
                            [('node_in_1', 'quantize'),
                             ('node_in_2', 'quantize'),
                             ('node_in_3', 'quantize'),
                             ('node_in_4', 'quantize'),
                             ('node_in_5', 'quantize'),
                             ('quantize', 'node_out_1'),
                             ('node_out_1', 'op_output')
                             ],
                            {
                                'node_out_1': {
                                    'shape': None,
                                    'value': None,
                                },
                                'node_in_1': {
                                    'shape': np.array([4]),
                                    'value': np.array([5, 17, 0, 100], dtype=np.float32),
                                },
                                'node_in_2': {
                                    'shape': np.array([4]),
                                    'value': np.array([0, 12, 12, 12], dtype=np.float32),
                                },
                                'node_in_3': {
                                    'shape': np.array([4]),
                                    'value': np.array([10, 20, 20, 20], dtype=np.float32),
                                },
                                'node_in_4': {
                                    'shape': np.array([4]),
                                    'value': np.array([0, 0, 0, 0], dtype=np.float32),
                                },
                                'node_in_5': {
                                    'shape': np.array([4]),
                                    'value': np.array([1, 1, 1, 1], dtype=np.float32),
                                },
                            })

        exp_node = Node(graph, 'quantize')
        FakeQuantize.infer(exp_node)
        quantize_shape = np.array([4])
        quantize_value = np.array([1, 1, 0, 1], dtype=np.float32)
        res_shape = graph.node['node_out_1']['shape']
        res_value = graph.node['node_out_1']['value']
        for i in range(0, len(quantize_shape)):
            self.assertEqual(quantize_shape[i], res_shape[i])
        for i in range(0, len(quantize_value)):
            self.assertAlmostEqual(quantize_value[i], res_value[i], places=6)
