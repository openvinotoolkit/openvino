"""
 Copyright (c) 2018 Intel Corporation

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

from extensions.front.eltwise_n import EltwiseNReplacement
from mo.utils.unittest.graph import build_graph
from mo.graph.graph import Node


class TestAddNFrontReplacement(unittest.TestCase):
    def test_replase_eltwise_n(self):
        graph = build_graph(
            {'node_1': {'type': 'Identity', 'value': None, 'kind': 'op', 'op': 'Placeholder'},
             'node_2': {'type': 'Identity', 'value': None, 'kind': 'op'},
             'node_3': {'type': 'Identity', 'value': None, 'kind': 'op'},
             'add_n': {'value': None, 'operation': 'sum', 'type': None, 'kind': 'op', 'op': 'EltwiseN'},
             'node_4': {'type': 'Identity', 'value': None, 'kind': 'op'},
             },
            [('node_1', 'node_2'),
             ('node_2', 'add_n'),
             ('node_3', 'add_n'),
             ('add_n', 'node_4'), ],
        )

        add_n_node = Node(graph, 'add_n')
        rep_op = EltwiseNReplacement()
        rep_op.replace_op(graph, add_n_node)
        eltwise_nodes = [node for node, attrs in list(graph.nodes(data=True)) if attrs['type'] == 'Eltwise']
        self.assertEqual(len(eltwise_nodes), 1)

    def test_replase_eltwise_n_2(self):
        graph = build_graph(
            {'node_1': {'type': 'Identity', 'value': None, 'kind': 'op', 'op': 'Placeholder'},
             'node_2': {'type': 'Identity', 'value': None, 'kind': 'op'},
             'node_3': {'type': 'Identity', 'value': None, 'kind': 'op'},
             'node_4': {'type': 'Identity', 'value': None, 'kind': 'op'},
             'add_n': {'value': None, 'operation': 'sum', 'type': None, 'kind': 'op', 'op': 'EltwiseN'},
             'node_5': {'type': 'Identity', 'value': None, 'kind': 'op'},
             },
            [('node_1', 'node_2'),
             ('node_2', 'add_n'),
             ('node_3', 'add_n'),
             ('node_4', 'add_n'),
             ('add_n', 'node_5'), ],
        )

        add_n_node = Node(graph, 'add_n')
        rep_op = EltwiseNReplacement()
        rep_op.replace_op(graph, add_n_node)
        eltwise_nodes = [node for node, attrs in list(graph.nodes(data=True)) if attrs['type'] == 'Eltwise']
        self.assertEqual(len(eltwise_nodes), 2)

    def test_replase_eltwise_n_3(self):
        graph = build_graph(
            {'node_1': {'type': 'Identity', 'value': None, 'kind': 'op', 'op': 'Placeholder'},
             'node_2': {'type': 'Identity', 'value': None, 'kind': 'op'},
             'node_3': {'type': 'Identity', 'value': None, 'kind': 'op'},
             'node_4': {'type': 'Identity', 'value': None, 'kind': 'op'},
             'node_5': {'type': 'Identity', 'value': None, 'kind': 'op'},
             'add_n': {'value': None, 'operation': 'sum', 'type': None, 'kind': 'op', 'op': 'EltwiseN'},
             'node_6': {'type': 'Identity', 'value': None, 'kind': 'op'},
             },
            [('node_1', 'node_2'),
             ('node_2', 'add_n'),
             ('node_3', 'add_n'),
             ('node_4', 'add_n'),
             ('node_5', 'add_n'),
             ('add_n', 'node_6'), ],
        )

        add_n_node = Node(graph, 'add_n')
        rep_op = EltwiseNReplacement()
        rep_op.replace_op(graph, add_n_node)
        eltwise_nodes = [node for node, attrs in list(graph.nodes(data=True)) if attrs['type'] == 'Eltwise']
        self.assertEqual(len(eltwise_nodes), 3)
