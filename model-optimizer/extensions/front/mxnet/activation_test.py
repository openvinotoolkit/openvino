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

from extensions.front.mxnet.activation import ActivationFrontExtractor
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph


class TestActivationFrontExtractorOp(unittest.TestCase):
    def test_extract_sigmoid_layer(self):
        graph = build_graph(
            {'node_1': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
             'act_node': {'type': 'Activation', 'kind': 'op', 'op': 'Activation', },
             'node_2': {'type': 'Identity', 'kind': 'op'},
             },
            [
                ('node_1', 'act_node'),
                ('act_node', 'node_2'),
            ],
            {
                'act_node': {'symbol_dict': {'attrs': {'act_type': 'sigmoid'}}},
            })

        act_node = Node(graph, 'act_node')
        act_extr_op = ActivationFrontExtractor()
        supported = act_extr_op.extract(act_node)
        self.assertTrue(supported)
        self.assertEqual(act_node['op'], 'Sigmoid')

    def test_extract_relu_layer(self):
        graph = build_graph(
            {'node_1': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
             'act_node': {'type': 'relu', 'kind': 'op', 'op': 'Activation', },
             'node_2': {'type': 'Identity', 'kind': 'op'},
             },
            [
                ('node_1', 'act_node'),
                ('act_node', 'node_2'),
            ],
            {
                'act_node': {'symbol_dict': {'attrs': {'act_type': 'relu'}}},
            })

        act_node = Node(graph, 'act_node')
        act_extr_op = ActivationFrontExtractor()
        supported = act_extr_op.extract(act_node)
        self.assertTrue(supported)
        self.assertEqual(act_node['op'], 'ReLU')
