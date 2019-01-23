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

from mo.graph.graph import erase_node, get_node_id_by_name, Node, replace_node, get_inputs_with_ports
from mo.ops.const import Const
from mo.utils.error import Error
from mo.utils.unittest.graph import build_graph, compare_graphs

nodes = {
    '0': {'name': 'input1', 'type': 'Identity', 'value': None, 'kind': 'op', 'op': 'Placeholder'},
    '1': {'name': 'input2', 'type': 'Identity', 'value': None, 'kind': 'op', 'op': 'Placeholder'},
    '2': {'name': 'node_1', 'type': 'Identity', 'value': None, 'kind': 'op', 'op': 'NotPlaceholder'},
    '3': {'name': 'node_2', 'type': 'Identity', 'value': None, 'kind': 'op', 'op': 'NotPlaceholder'},
    '4': {'name': 'node_3', 'type': 'Identity', 'value': None, 'kind': 'op', 'op': 'NotPlaceholder'},
    '5': {'name': 'node_4', 'type': 'Identity', 'value': None, 'kind': 'op', 'op': 'NotPlaceholder'},
    '6': {'name': 'output', 'type': 'Identity', 'value': None, 'kind': 'op', 'op': 'OpOutput',
          'is_output': True},
    'input_3': {'name': 'input_3', 'type': 'Identity', 'value': None, 'kind': 'op', 'op': 'Placeholder'}
}
edges = {
    ('0', '2'),
    ('2', '3'),
    ('4', '6'),
    ('1', '5'),
    ('5', '6'),
    ('input_3', '6')
}


class TestGetNodeById(unittest.TestCase):
    def setUp(self):
        self.graph = build_graph(nodes, edges)

    def test_get_node_id_by_name(self):
        self.assertEqual(get_node_id_by_name(self.graph, 'input1'), '0')

    def test_get_node_id_by_name_1(self):
        self.assertEqual(get_node_id_by_name(self.graph, 'input2'), '1')

    def test_get_node_id_by_name_2(self):
        self.assertEqual(get_node_id_by_name(self.graph, 'node_1'), '2')

    def test_get_node_id_by_name_3(self):
        self.assertEqual(get_node_id_by_name(self.graph, 'node_2'), '3')

    def test_get_node_id_by_name_4(self):
        self.assertEqual(get_node_id_by_name(self.graph, 'node_3'), '4')

    def test_get_node_id_by_name_5(self):
        self.assertEqual(get_node_id_by_name(self.graph, 'node_4'), '5')

    def test_get_node_id_by_name_6(self):
        self.assertEqual(get_node_id_by_name(self.graph, 'output'), '6')

    def test_get_node_id_by_name_7(self):
        self.assertEqual(get_node_id_by_name(self.graph, 'input_3'), 'input_3')

    def test_get_node_id_by_name_8(self):
        self.assertRaises(Error, get_node_id_by_name, self.graph, '1')


class TestEraseNode(unittest.TestCase):
    def test_remove_noop_nodes_middle(self):
        graph = build_graph(
            {
                'input': {'type': 'Placeholder', 'value': None, 'kind': 'op'},
                'noop': {'type': 'NoOp', 'value': None, 'kind': 'op'},
                'output': {'type': 'Identity', 'value': None, 'kind': 'op'},
            },
            [('input', 'noop'),
             ('noop', 'output')])

        self.assertEqual(len(graph.nodes()), 3)
        self.assertEqual(len(graph.edges()), 2)
        self.assertListEqual(list(graph.out_edges('input')), [('input', 'noop')])

        erase_node(Node(graph, 'noop'))

        self.assertEqual(len(graph.nodes()), 2)
        self.assertEqual(len(graph.edges()), 1)
        self.assertListEqual(list(graph.out_edges('input')), [('input', 'output')])

    def test_remove_noop_nodes_middle_2(self):
        graph = build_graph(
            {
                'input': {'type': 'Placeholder', 'value': None, 'kind': 'op'},
                'noop': {'type': 'NoOp', 'value': None, 'kind': 'op'},
                'output_1': {'type': 'Identity', 'value': None, 'kind': 'op'},
                'output_2': {'type': 'Identity', 'value': None, 'kind': 'op'},
                'output_3': {'type': 'Identity', 'value': None, 'kind': 'op'},
            },
            [('input', 'noop'),
             ('noop', 'output_1', {'in': 4, 'out': 0}),
             ('noop', 'output_2', {'in': 2, 'out': 0}),
             ('noop', 'output_3', {'in': 10, 'out': 0})])

        ref_graph = build_graph(
            {
                'input': {'type': 'Placeholder', 'value': None, 'kind': 'op'},
                'output_1': {'type': 'Identity', 'value': None, 'kind': 'op'},
                'output_2': {'type': 'Identity', 'value': None, 'kind': 'op'},
                'output_3': {'type': 'Identity', 'value': None, 'kind': 'op'},
            },
            [('input', 'output_1', {'in': 4, 'out': 0}),
             ('input', 'output_2', {'in': 2, 'out': 0}),
             ('input', 'output_3', {'in': 10, 'out': 0})],
            nodes_with_edges_only=True)

        erase_node(Node(graph, 'noop'))

        compare_graphs(graph, ref_graph, 'output_1')

    def test_remove_noop_nodes_check_out_port(self):
        graph = build_graph(
            {
                'input': {'type': 'Placeholder', 'value': None, 'kind': 'op'},
                'noop': {'type': 'NoOp', 'value': None, 'kind': 'op'},
                'output_1': {'type': 'Identity', 'value': None, 'kind': 'op'},
                'output_2': {'type': 'Identity', 'value': None, 'kind': 'op'},
                'output_3': {'type': 'Identity', 'value': None, 'kind': 'op'},
            },
            [('input', 'noop'),
             ('noop', 'output_1', {'in': 4, 'out': 1}),
             ('noop', 'output_2', {'in': 2, 'out': 1}),
             ('noop', 'output_3', {'in': 10, 'out': 1})])

        ref_graph = build_graph(
            {
                'input': {'type': 'Placeholder', 'value': None, 'kind': 'op'},
                'output_1': {'type': 'Identity', 'value': None, 'kind': 'op'},
                'output_2': {'type': 'Identity', 'value': None, 'kind': 'op'},
                'output_3': {'type': 'Identity', 'value': None, 'kind': 'op'},
            },
            [('input', 'output_1', {'in': 4, 'out': 0}),
             ('input', 'output_2', {'in': 2, 'out': 0}),
             ('input', 'output_3', {'in': 10, 'out': 0})],
            nodes_with_edges_only=True)

        erase_node(Node(graph, 'noop'))

        compare_graphs(graph, ref_graph, 'output_1')

    def test_remove_noop_nodes_too_many_outputs(self):
        graph = build_graph(
            {
                'input': {'type': 'Placeholder', 'value': None, 'kind': 'op'},
                'noop': {'type': 'NoOp', 'value': None, 'kind': 'op'},
                'output_1': {'type': 'Identity', 'value': None, 'kind': 'op'},
                'output_2': {'type': 'Identity', 'value': None, 'kind': 'op'},
                'output_3': {'type': 'Identity', 'value': None, 'kind': 'op'},
            },
            [('input', 'noop'),
             ('noop', 'output_1', {'in': 4, 'out': 0}),
             ('noop', 'output_2', {'in': 2, 'out': 1}),
             ('noop', 'output_3', {'in': 10, 'out': 0})])

        self.assertRaises(AssertionError, erase_node, Node(graph, 'noop'))

    def test_remove_noop_nodes_front(self):
        graph = build_graph(
            {
                'noop': {'type': 'NoOp', 'value': None, 'kind': 'op'},
                'output': {'type': 'Identity', 'value': None, 'kind': 'op'}
            },
            [('noop', 'output')]
        )

        self.assertEqual(len(graph.nodes()), 2)
        self.assertEqual(len(graph.edges()), 1)
        self.assertListEqual(list(graph.out_edges('noop')), [('noop', 'output')])

        erase_node(Node(graph, 'noop'))

        self.assertEqual(len(graph.nodes()), 1)
        self.assertEqual(len(graph.edges()), 0)
        self.assertEqual(len(graph.in_edges('output')), 0)

    def test_remove_noop_nodes_back(self):
        graph = build_graph(
            {
                'input': {'type': 'Placeholder', 'value': None, 'kind': 'op'},
                'noop': {'type': 'NoOp', 'value': None, 'kind': 'op'}
            },
            [('input', 'noop')]
        )

        self.assertEqual(len(graph.nodes()), 2)
        self.assertEqual(len(graph.edges()), 1)
        self.assertListEqual(list(graph.in_edges('noop')), [('input', 'noop')])

        erase_node(Node(graph, 'noop'))

        self.assertEqual(len(graph.nodes()), 1)
        self.assertEqual(len(graph.edges()), 0)
        self.assertEqual(len(graph.in_edges('input')), 0)

    def test_remove_noop_nodes_noop_only(self):
        import networkx as nx
        graph = nx.MultiDiGraph()
        graph.add_node('noop', **{'type': 'NoOp', 'value': None, 'kind': 'op'})

        self.assertEqual(len(graph.nodes()), 1)
        self.assertEqual(len(graph.edges()), 0)

        erase_node(Node(graph, 'noop'))

        self.assertEqual(len(graph.nodes()), 0)
        self.assertEqual(len(graph.edges()), 0)

    def test_remove_noop_error(self):
        graph = build_graph(
            {
                'input_1': {'type': 'Placeholder', 'value': None, 'kind': 'op'},
                'input_2': {'type': 'Placeholder', 'value': None, 'kind': 'op'},
                'input_3': {'type': 'Placeholder', 'value': None, 'kind': 'op'},
                'noop': {'type': 'NoOp', 'value': None, 'kind': 'op'},
                'output_1': {'type': 'Identity', 'value': None, 'kind': 'op'},
                'output_2': {'type': 'Identity', 'value': None, 'kind': 'op'},
                'output_3': {'type': 'Identity', 'value': None, 'kind': 'op'},
            },
            [('input_1', 'noop'),
             ('input_2', 'noop'),
             ('input_3', 'noop'),
             ('noop', 'output_1'),
             ('noop', 'output_2'),
             ('noop', 'output_3')])
        self.assertRaises(AssertionError, erase_node, Node(graph, 'noop'))


class TestReplaceNode(unittest.TestCase):
    def test_replace_node_one_consumer(self):
        graph = build_graph(
            {
                'input_1': {'type': 'Placeholder', 'value': None, 'kind': 'op'},
                'input_2': {'type': 'Placeholder', 'value': None, 'kind': 'op'},
                'old': {'type': 'Identity', 'value': None, 'kind': 'op', 'is_output': True},
                'output': {'type': 'OpOutput', 'value': None, 'kind': 'op'},
            },
            [('input_1', 'old'),
             ('input_2', 'old'),
             ('old', 'output')])

        new_node = Const(graph, {'name': 'new'}).create_node([Node(graph, 'input_1'), Node(graph, 'input_2')])
        replace_node(Node(graph, 'old'), new_node)

        self.assertEqual(len(graph.nodes()), 4)
        self.assertEqual(len(graph.edges()), 3)
        self.assertEqual(new_node['is_output'], True)
        self.assertListEqual(list(graph.out_edges('new')), [('new', 'output')])

    def test_replace_node_several_consumers(self):
        graph = build_graph(
            {
                'input_1': {'type': 'Placeholder', 'value': None, 'kind': 'op'},
                'input_2': {'type': 'Placeholder', 'value': None, 'kind': 'op'},
                'old': {'type': 'Identity', 'value': None, 'kind': 'op'},
                'output_1': {'type': 'Identity', 'value': None, 'kind': 'op'},
                'output_2': {'type': 'Identity', 'value': None, 'kind': 'op'},
                'output_3': {'type': 'Identity', 'value': None, 'kind': 'op'},
            },
            [('input_1', 'old'),
             ('input_2', 'old'),
             ('old', 'output_3'),
             ('old', 'output_2'),
             ('old', 'output_1'),
             ])

        new_node = Const(graph, {'name': 'new'}).create_node([Node(graph, 'input_1'), Node(graph, 'input_2')])
        replace_node(Node(graph, 'old'), new_node)

        self.assertEqual(len(graph.nodes()), 6)
        self.assertEqual(len(graph.edges()), 5)
        self.assertListEqual(sorted(graph.out_edges('new')), [('new', 'output_1'), ('new', 'output_2'),
                                                              ('new', 'output_3')])
        expected_result = [('new', 'output_1', {'in': 0, 'out': 2, 'name': 'old'}),
                           ('new', 'output_2', {'in': 0, 'out': 1, 'name': 'old'}),
                           ('new', 'output_3', {'in': 0, 'out': 0, 'name': 'old'})]
        self.assertListEqual(sorted(graph.out_edges('new', data=True)), expected_result)


class GetNodesWithPorts(unittest.TestCase):
    def test_get_nodes_with_ports(self):
        nodes = {
            'one': {},
            'two': {},
            'three': {},
            'four': {},
            'five': {}
        }
        edges = [
            ('one', 'two', {'in': 0, 'out': 0}),
            ('two', 'three', {'in': 0, 'out': 0}),
            ('two', 'four', {'in': 0, 'out': 1}),
            ('two', 'five', {'in': 0, 'out': 2}),
            ('three', 'five', {'in': 1, 'out': 0})
        ]
        graph = build_graph(nodes, edges)
        match = {
            'one': Node(graph, 'one'),
            'two': Node(graph, 'two'),
            'three': Node(graph, 'three'),
            'four': Node(graph, 'four'),
            'five': Node(graph, 'five'),

        }
        input_names_in_pattern = ['one', 'three']
        result = get_inputs_with_ports(graph=graph, match=match, pattern_edges=edges,
                                       input_names_in_pattern=input_names_in_pattern)
        self.assertListEqual([(match['one'], 0), (match['three'], 0)], result)
