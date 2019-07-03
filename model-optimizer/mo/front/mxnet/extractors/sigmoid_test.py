"""
 Copyright (c) 2018-2019 Intel Corporation

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

from mo.front.mxnet.extractors.sigmoid import Sigmoid
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph


class TestSigmoidFrontExtractorOp(unittest.TestCase):
    def test_extract_sigmoid_layer(self):
        graph = build_graph(
            {'node_1': {'type': 'Placeholder', 'kind': 'op', 'op': 'Placeholder'},
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
        sigmoid_extr_op = Sigmoid()
        supported = sigmoid_extr_op.extract(sigmoid_node)
        self.assertTrue(supported)
        self.assertEqual(sigmoid_node['op'], 'Activation')
        self.assertEqual(sigmoid_node['operation'], 'sigmoid')
