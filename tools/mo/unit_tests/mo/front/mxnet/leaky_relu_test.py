# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.front.mxnet.leaky_relu import LeakyReLUFrontExtractor
from openvino.tools.mo.graph.graph import Node
from unit_tests.utils.graph import build_graph


class TestLeakyReLUFrontExtractorOp(unittest.TestCase):
    def test_extract_leaky_relu_layer(self):
        graph = build_graph(
            {'node_1': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
             'act_node': {'type': 'LeakyReLU', 'kind': 'op', 'op': 'LeakyReLU', },
             'node_2': {'type': 'Identity', 'kind': 'op'},
             },
            [
                ('node_1', 'act_node'),
                ('act_node', 'node_2'),
            ],
            {
                'act_node': {'symbol_dict': {'attrs': {'slope': '0.6'}}},
            })

        act_node = Node(graph, 'act_node')
        act_extr_op = LeakyReLUFrontExtractor()
        supported = act_extr_op.extract(act_node)
        self.assertTrue(supported)
        self.assertEqual(act_node['op'], 'LeakyReLU')
        self.assertEqual(act_node['negative_slope'], 0.6)

    def test_extract_prelu_layer(self):
        graph = build_graph(
            {'node_1': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
             'node_3': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
             'act_node': {'type': 'LeakyReLU', 'kind': 'op', 'op': 'LeakyReLU', },
             'node_2': {'type': 'Identity', 'kind': 'op'},
             },
            [
                ('node_1', 'act_node'),
                ('node_3', 'act_node'),
                ('act_node', 'node_2'),
            ],
            {
                'act_node': {'symbol_dict': {'attrs': {'act_type': 'prelu'}}},
                'node_3': {'value': np.array([1], dtype=np.float32)},
            })
        act_node = Node(graph, 'act_node')
        act_extr_op = LeakyReLUFrontExtractor()
        supported = act_extr_op.extract(act_node)
        self.assertTrue(supported)
        self.assertEqual(act_node['op'], 'PReLU')

    def test_extract_elu_layer(self):
        graph = build_graph(
            {'node_1': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
             'act_node': {'type': 'LeakyReLU', 'kind': 'op', 'op': 'LeakyReLU', },
             'node_2': {'type': 'Parameter', 'kind': 'op'},
             },
            [
                ('node_1', 'act_node'),
                ('act_node', 'node_2'),
            ],
            {
                'act_node': {'symbol_dict': {'attrs': {'act_type': 'elu'}}},
            })

        act_node = Node(graph, 'act_node')
        act_extr_op = LeakyReLUFrontExtractor()
        supported = act_extr_op.extract(act_node)
        self.assertTrue(supported)
        self.assertEqual(act_node['op'], 'Elu')
