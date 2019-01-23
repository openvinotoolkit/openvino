"""
 Copyright (c) 2017-2018 Intel Corporation

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

from extensions.front.mxnet.check_softmax_node_inputs import CheckSoftmaxNodeInputs
from mo.utils.unittest.graph import build_graph
from mo.graph.graph import Node


class TestCheckSoftmaxNodeInputs(unittest.TestCase):
    def test_remove_softmax_output_input(self):
        graph = build_graph(
            {'node_1': {'type': 'Identity', 'value': None, 'kind': 'op', 'op': 'Placeholder'},
             'node_2': {'type': 'Identity', 'value': None, 'kind': 'op', 'op': 'Placeholder'},
             'softmax': {'type': 'SoftmaxOutput', 'value': None, 'kind': 'op', 'op': 'SoftmaxOutput'},
             },
            [('node_1', 'softmax'),
             ('node_2', 'softmax')
             ])

        pattern = CheckSoftmaxNodeInputs()
        pattern.find_and_replace_pattern(graph)

        node_softmax = Node(graph, 'softmax')

        self.assertEqual(len(node_softmax.in_nodes()), 1)

        node_input1 = node_softmax.in_node(0)
        self.assertEqual(node_input1.name, 'node_1')

    def test_remove_softmax_activation_input(self):
        graph = build_graph(
            {'node_1': {'type': 'Identity', 'value': None, 'kind': 'op', 'op': 'Placeholder'},
             'softmax': {'type': 'SoftmaxActivation', 'value': None, 'kind': 'op', 'op': 'SoftmaxActivation'},
             },
            [('node_1', 'softmax')])

        pattern = CheckSoftmaxNodeInputs()
        pattern.find_and_replace_pattern(graph)

        node_softmax = Node(graph, 'softmax')

        self.assertEqual(len(node_softmax.in_nodes()), 1)

        node_input1 = node_softmax.in_node(0)
        self.assertEqual(node_input1.name, 'node_1')
