# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.front.mxnet.check_softmax_node_inputs import CheckSoftmaxNodeInputs
from openvino.tools.mo.graph.graph import Node
from unit_tests.utils.graph import build_graph


class TestCheckSoftmaxNodeInputs(unittest.TestCase):
    def test_remove_softmax_output_input(self):
        graph = build_graph(
            {'node_1': {'type': 'Identity', 'value': None, 'kind': 'op', 'op': 'Parameter'},
             'node_2': {'type': 'Identity', 'value': None, 'kind': 'op', 'op': 'Parameter'},
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
            {'node_1': {'type': 'Identity', 'value': None, 'kind': 'op', 'op': 'Parameter'},
             'softmax': {'type': 'SoftmaxActivation', 'value': None, 'kind': 'op', 'op': 'SoftmaxActivation'},
             },
            [('node_1', 'softmax')])

        pattern = CheckSoftmaxNodeInputs()
        pattern.find_and_replace_pattern(graph)

        node_softmax = Node(graph, 'softmax')

        self.assertEqual(len(node_softmax.in_nodes()), 1)

        node_input1 = node_softmax.in_node(0)
        self.assertEqual(node_input1.name, 'node_1')
