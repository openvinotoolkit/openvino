# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.front.mxnet.extractors.relu import ReLUFrontExtractor
from openvino.tools.mo.graph.graph import Node
from unit_tests.utils.graph import build_graph


class TestReluFrontExtractorOp(unittest.TestCase):
    def test_extract_relu_layer(self):
        graph = build_graph(
            {'node_1': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
             'relu_node': {'type': 'relu', 'kind': 'op', 'op': 'relu', },
             'node_2': {'type': 'Parameter', 'kind': 'op'},
             },
            [
                ('node_1', 'relu_node'),
                ('relu_node', 'node_2'),
            ],
            {
                'relu_node': {'symbol_dict': {'attrs': {}}},
            })

        relu_node = Node(graph, 'relu_node')
        relu_extr_op = ReLUFrontExtractor()
        supported = relu_extr_op.extract(relu_node)
        self.assertTrue(supported)
        self.assertEqual(relu_node['op'], 'ReLU')
