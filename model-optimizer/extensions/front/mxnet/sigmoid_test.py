# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from extensions.front.mxnet.sigmoid import SigmoidFrontExtractor
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph


class TestSigmoidFrontExtractorOp(unittest.TestCase):
    def test_extract_sigmoid_layer(self):
        graph = build_graph(
            {'node_1': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
             'sigmoid_node': {'type': 'sigmoid', 'kind': 'op', 'op': 'sigmoid', },
             'node_3': {'type': 'Identity', 'value': None, 'kind': 'op'},
             },
            [
                ('node_1', 'sigmoid_node'),
                ('sigmoid_node', 'node_3'),
            ],
            {
                'sigmoid_node': {'symbol_dict': {'attrs': {}}},
            })

        sigmoid_node = Node(graph, 'sigmoid_node')
        sigmoid_extr_op = SigmoidFrontExtractor
        supported = sigmoid_extr_op.extract(sigmoid_node)
        self.assertTrue(supported)
        self.assertEqual(sigmoid_node['op'], 'Sigmoid')
