# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
from generator import generator, generate

from mo.graph.graph import Node, Graph, add_opoutput, dict_includes_compare_attrs, get_edge_attribute_between_nodes, \
    set_edge_attribute_between_nodes
from mo.ops.const import Const
from mo.utils.error import Error
from mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph

nodes = {
    '0': {'name': 'input1', 'type': 'Identity', 'value': None, 'kind': 'op', 'op': 'Parameter'},
    '1': {'name': 'input2', 'type': 'Identity', 'value': None, 'kind': 'op', 'op': 'Parameter'},
    '2': {'name': 'node_1', 'type': 'Identity', 'value': None, 'kind': 'op', 'op': 'NotPlaceholder'},
    '3': {'name': 'node_2', 'type': 'Identity', 'value': None, 'kind': 'op', 'op': 'NotPlaceholder'},
    '4': {'name': 'node_3', 'type': 'Identity', 'value': None, 'kind': 'op', 'op': 'NotPlaceholder'},
    '5': {'name': 'node_4', 'type': 'Identity', 'value': None, 'kind': 'op', 'op': 'NotPlaceholder'},
    '6': {'name': 'output', 'value': None, 'kind': 'op', 'op': 'Result'},
    'input_3': {'name': 'input_3', 'type': 'Identity', 'value': None, 'kind': 'op', 'op': 'Parameter'}
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
        self.assertEqual(self.graph.get_node_id_by_name('input1'), '0')

    def test_get_node_id_by_name_1(self):
        self.assertEqual(self.graph.get_node_id_by_name('input2'), '1')

    def test_get_node_id_by_name_2(self):
        self.assertEqual(self.graph.get_node_id_by_name('node_1'), '2')

    def test_get_node_id_by_name_3(self):
        self.assertEqual(self.graph.get_node_id_by_name('node_2'), '3')

    def test_get_node_id_by_name_4(self):
        self.assertEqual(self.graph.get_node_id_by_name('node_3'), '4')

    def test_get_node_id_by_name_5(self):
        self.assertEqual(self.graph.get_node_id_by_name('node_4'), '5')

    def test_get_node_id_by_name_6(self):
        self.assertEqual(self.graph.get_node_id_by_name('output'), '6')

    def test_get_node_id_by_name_7(self):
        self.assertEqual(self.graph.get_node_id_by_name('input_3'), 'input_3')

    def test_get_node_id_by_name_8(self):
        self.assertRaises(Error, self.graph.get_node_id_by_name, '1')


class TestEraseNode(unittest.TestCase):
    def test_remove_noop_nodes_middle(self):
        graph = build_graph(
            {
                'input': {'type': 'Parameter', 'value': None, 'kind': 'op'},
                'noop': {'type': 'NoOp', 'value': None, 'kind': 'op'},
                'output': {'type': 'Identity', 'value': None, 'kind': 'op'},
            },
            [('input', 'noop'),
             ('noop', 'output')])

        self.assertEqual(len(graph.nodes()), 3)
        self.assertEqual(len(graph.edges()), 2)
        self.assertListEqual(list(graph.out_edges('input')), [('input', 'noop')])

        graph.erase_node(Node(graph, 'noop'))

        self.assertEqual(len(graph.nodes()), 2)
        self.assertEqual(len(graph.edges()), 1)
        self.assertListEqual(list(graph.out_edges('input')), [('input', 'output')])

    def test_remove_noop_nodes_middle_2(self):
        graph = build_graph(
            {
                'input': {'type': 'Parameter', 'value': None, 'kind': 'op'},
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
                'input': {'type': 'Parameter', 'value': None, 'kind': 'op'},
                'output_1': {'type': 'Identity', 'value': None, 'kind': 'op'},
                'output_2': {'type': 'Identity', 'value': None, 'kind': 'op'},
                'output_3': {'type': 'Identity', 'value': None, 'kind': 'op'},
            },
            [('input', 'output_1', {'in': 4, 'out': 0}),
             ('input', 'output_2', {'in': 2, 'out': 0}),
             ('input', 'output_3', {'in': 10, 'out': 0})],
            nodes_with_edges_only=True)

        graph.erase_node(Node(graph, 'noop'))

        compare_graphs(graph, ref_graph, 'output_1')

    def test_remove_noop_nodes_check_out_port(self):
        graph = build_graph(
            {
                'input': {'type': 'Parameter', 'value': None, 'kind': 'op'},
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
                'input': {'type': 'Parameter', 'value': None, 'kind': 'op'},
                'output_1': {'type': 'Identity', 'value': None, 'kind': 'op'},
                'output_2': {'type': 'Identity', 'value': None, 'kind': 'op'},
                'output_3': {'type': 'Identity', 'value': None, 'kind': 'op'},
            },
            [('input', 'output_1', {'in': 4, 'out': 0}),
             ('input', 'output_2', {'in': 2, 'out': 0}),
             ('input', 'output_3', {'in': 10, 'out': 0})],
            nodes_with_edges_only=True)

        graph.erase_node(Node(graph, 'noop'))

        compare_graphs(graph, ref_graph, 'output_1')

    def test_remove_noop_nodes_too_many_outputs(self):
        graph = build_graph(
            {
                'input': {'type': 'Parameter', 'value': None, 'kind': 'op'},
                'noop': {'type': 'NoOp', 'value': None, 'kind': 'op'},
                'output_1': {'type': 'Identity', 'value': None, 'kind': 'op'},
                'output_2': {'type': 'Identity', 'value': None, 'kind': 'op'},
                'output_3': {'type': 'Identity', 'value': None, 'kind': 'op'},
            },
            [('input', 'noop'),
             ('noop', 'output_1', {'in': 4, 'out': 0}),
             ('noop', 'output_2', {'in': 2, 'out': 1}),
             ('noop', 'output_3', {'in': 10, 'out': 0})])

        self.assertRaises(AssertionError, graph.erase_node, Node(graph, 'noop'))

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

        graph.erase_node(Node(graph, 'noop'))

        self.assertEqual(len(graph.nodes()), 1)
        self.assertEqual(len(graph.edges()), 0)
        self.assertEqual(len(graph.in_edges('output')), 0)

    def test_remove_noop_nodes_back(self):
        graph = build_graph(
            {
                'input': {'type': 'Parameter', 'value': None, 'kind': 'op'},
                'noop': {'type': 'NoOp', 'value': None, 'kind': 'op'}
            },
            [('input', 'noop')]
        )

        self.assertEqual(len(graph.nodes()), 2)
        self.assertEqual(len(graph.edges()), 1)
        self.assertListEqual(list(graph.in_edges('noop')), [('input', 'noop')])

        graph.erase_node(Node(graph, 'noop'))

        self.assertEqual(len(graph.nodes()), 1)
        self.assertEqual(len(graph.edges()), 0)
        self.assertEqual(len(graph.in_edges('input')), 0)

    def test_remove_noop_nodes_noop_only(self):
        graph = Graph()
        graph.add_node('noop', **{'type': 'NoOp', 'value': None, 'kind': 'op'})

        self.assertEqual(len(graph.nodes()), 1)
        self.assertEqual(len(graph.edges()), 0)

        graph.erase_node(Node(graph, 'noop'))

        self.assertEqual(len(graph.nodes()), 0)
        self.assertEqual(len(graph.edges()), 0)

    def test_remove_noop_error(self):
        graph = build_graph(
            {
                'input_1': {'type': 'Parameter', 'value': None, 'kind': 'op'},
                'input_2': {'type': 'Parameter', 'value': None, 'kind': 'op'},
                'input_3': {'type': 'Parameter', 'value': None, 'kind': 'op'},
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
        self.assertRaises(AssertionError, graph.erase_node, Node(graph, 'noop'))


class TestReplaceNode(unittest.TestCase):
    def test_replace_node_one_consumer(self):
        graph = build_graph(
            {
                'input_1': {'type': 'Parameter', 'value': None, 'kind': 'op'},
                'input_2': {'type': 'Parameter', 'value': None, 'kind': 'op'},
                'old': {'type': 'Identity', 'value': None, 'kind': 'op'},
                'output': {'op': 'Result', 'value': None, 'kind': 'op'},
            },
            [('input_1', 'old'),
             ('input_2', 'old'),
             ('old', 'output')])

        new_node = Const(graph, {'name': 'new'}).create_node([Node(graph, 'input_1'), Node(graph, 'input_2')])

        old_node = Node(graph, 'old')
        old_node.replace_node(new_node)

        self.assertEqual(len(graph.nodes()), 4)
        self.assertEqual(len(graph.edges()), 3)
        self.assertEqual(new_node.out_node().op, 'Result')
        self.assertEqual(len(graph.out_edges('new')), 1)

    def test_replace_node_several_consumers(self):
        graph = build_graph(
            {
                'input_1': {'type': 'Parameter', 'value': None, 'kind': 'op'},
                'input_2': {'type': 'Parameter', 'value': None, 'kind': 'op'},
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
        Node(graph, 'old').replace_node(new_node)

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
        result = graph.get_inputs_with_ports(match=match, pattern_edges=edges,
                                       input_names_in_pattern=input_names_in_pattern)
        self.assertListEqual([(match['one'], 0), (match['three'], 0)], result)


class TestGraphShapeChecker(unittest.TestCase):
    nodes = {
        '0': {'type': 'Parameter', 'value': None, 'kind': 'op', 'op': 'Parameter'},
        '0_data': {'value': None, 'shape': None, 'kind': 'data'},

        '1': {'type': 'Parameter', 'value': None, 'kind': 'op', 'op': 'Parameter'},
        '1_data': {'value': None, 'shape': None, 'kind': 'data'},

        '2': {'type': 'Parameter', 'value': None, 'kind': 'op', 'op': 'Parameter'},
        '2_data': {'value': None, 'shape': None, 'kind': 'data'},
    }

    def test_check_shape_consistency_1(self):
        # No shape attr in data node
        graph = build_graph(self.nodes, [
            ('0', '0_data'),
            ('1', '1_data'),
            ('2', '2_data'),

            ('0_data', '1'),
            ('0_data', '2')
        ])

        del graph.node['2_data']['shape']

        with self.assertRaisesRegex(Error, r"Graph contains data nodes \(1\) with inconsistent shapes:.*"):
            graph.check_shapes_consistency()

    def test_check_shape_consistency_2(self):
        # No shape attr in data node
        graph = build_graph(self.nodes, [
            ('0', '0_data'),
            ('1', '1_data'),
            ('2', '2_data'),

            ('0_data', '1'),
            ('0_data', '2')
        ])

        graph.node['1_data']['shape'] = (1, 2, 3)
        graph.node['2_data']['shape'] = (1, 2, 3)

        with self.assertRaisesRegex(Error, r"Graph contains data nodes \(2\) with inconsistent shapes:.*"):
            graph.check_shapes_consistency()


@generator
class TestGraphPortsChecker(unittest.TestCase):
    nodes = {
        '0': {'type': 'Parameter', 'value': None, 'kind': 'op', 'op': 'Parameter'},
        '0_data': {'value': None, 'shape': None, 'kind': 'data'},

        '1': {'type': 'Parameter', 'value': None, 'kind': 'op', 'op': 'Parameter'},
        '1_data': {'value': None, 'shape': None, 'kind': 'data'},

        '2': {'type': 'Parameter', 'value': None, 'kind': 'op', 'op': 'Parameter'},
        '2_data': {'value': None, 'shape': None, 'kind': 'data'},

        '3': {'type': 'Parameter', 'value': None, 'kind': 'op', 'op': 'Parameter'},
        '3_data': {'value': None, 'shape': None, 'kind': 'data'},
    }

    @generate(*[('0', 'in', 1), ('0', 'out', 2), ('1', 'in', 2), ('3', 'out', 2)])
    def test_check_shape_consistency_1(self, node_id: str, port_type: str, port_idx: int):
        #
        #               ,->2-->2_data---,->3-->3_data
        #   0-->0_data-/-->1-->1_data--/
        #
        graph = build_graph(self.nodes, [
            ('0', '0_data'),
            ('1', '1_data'),
            ('2', '2_data'),
            ('3', '3_data'),

            ('0_data', '1'),
            ('0_data', '2'),
            ('1_data', '3'),
            ('2_data', '3'),
        ])

        node = Node(graph, node_id)
        if port_type == 'in':
            node.add_input_port(idx=port_idx)
        else:
            node.add_output_port(idx=port_idx)

        with self.assertRaisesRegex(Error, "Node {} has not consecutive {} ports indexes:.*".format(node_id,
                                                                                                    port_type)):
            graph.check_nodes_ports_are_consecutive()


class TestNewGraphAPIMiddle(unittest.TestCase):

    nodes = {
        '0': {'type': 'Parameter', 'value': None, 'kind': 'op', 'op': 'Parameter'},
        '0_data': {'value': None, 'shape': None, 'kind': 'data'},

        '1': {'type': 'Parameter', 'value': None, 'kind': 'op', 'op': 'Parameter'},
        '1_data': {'value': None, 'shape': None, 'kind': 'data'},

        '2': {'type': 'Parameter', 'value': None, 'kind': 'op', 'op': 'Parameter'},
        '2_data': {'value': None, 'shape': None, 'kind': 'data'},

        '3': {'type': 'Parameter', 'value': None, 'kind': 'op', 'op': 'Parameter'},
        '3_data': {'value': None, 'shape': None, 'kind': 'data'},

        '4': {'type': 'Parameter', 'value': None, 'kind': 'op', 'op': 'Parameter'},
        '4_data': {'value': None, 'shape': None, 'kind': 'data'},

        'const_1': {'type': 'Const', 'value': None, 'kind': 'op', 'op': 'Const'},
        'const_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    }

      ###########################################
     ###### TESTS FOR PORT CLASS METHODS #######
    ###########################################

    def test_port_get_destinations_1(self):
        graph = build_graph(self.nodes, [
            ('0', '0_data'),
            ('1', '1_data'),
            ('2', '2_data'),

            ('0_data', '1'),
            ('0_data', '2')
        ])
        graph.__setattr__('stage', 'middle')

        node_0_out_port = Node(graph, '0').out_port(0)

        node_1_in_port = Node(graph, '1').in_port(0)
        node_2_in_port = Node(graph, '2').in_port(0)

        ports = node_0_out_port.get_destinations()

        self.assertTrue(len(ports) == 2)
        for port in ports:
            self.assertTrue(port in [node_1_in_port, node_2_in_port])

    def test_port_get_destination_1(self):
        graph = build_graph(self.nodes, [
            ('0', '0_data'),
            ('1', '1_data'),
            ('2', '2_data'),

            ('0_data', '1'),
            ('0_data', '2')
        ])
        graph.__setattr__('stage', 'middle')

        node_0_out_port = Node(graph, '0').out_port(0)

        node_1_in_port = Node(graph, '1').in_port(0)
        node_2_in_port = Node(graph, '2').in_port(0)

        with self.assertRaises(Error):
            node_0_out_port.get_destination()

    def test_port_get_destination_2(self):
        graph = build_graph(self.nodes, [
            ('0', '0_data'),
            ('1', '1_data'),
            ('0_data', '1'),
        ])
        graph.__setattr__('stage', 'middle')

        node_0_out_port = Node(graph, '0').out_port(0)

        node_1_in_port = Node(graph, '1').in_port(0)

        self.assertEqual(node_0_out_port.get_destination(), node_1_in_port)

    def test_port_get_source_1(self):
        graph = build_graph(self.nodes, [
            ('0', '0_data'),
            ('1', '1_data'),
            ('0_data', '1'),
        ])
        graph.__setattr__('stage', 'middle')

        node_0_out_port = Node(graph, '0').out_port(0)

        node_1_in_port = Node(graph, '1').in_port(0)

        self.assertEqual(node_1_in_port.get_source(), node_0_out_port)

    def test_port_get_source_2(self):
        graph = build_graph(self.nodes, [
            ('0', '0_data'),
            ('1', '1_data'),
            ('2', '2_data'),
            ('0_data', '1'),
            ('2_data', '1')
        ])
        graph.__setattr__('stage', 'middle')

        node_0 = Node(graph, '0')
        node_1 = Node(graph, '1')
        node_2 = Node(graph, '2')

        self.assertEqual(node_1.in_port(0).get_source(), node_0.out_port(0))
        self.assertEqual(node_1.in_port(1).get_source(), node_2.out_port(0))

    def test_port_get_source_3(self):
        graph = build_graph(self.nodes, [
            ('0', '0_data'),
            ('1', '1_data'),
            ('2', '2_data'),
        ])
        graph.__setattr__('stage', 'middle')

        node_0 = Node(graph, '0')
        node_1 = Node(graph, '1')
        node_2 = Node(graph, '2')

        node_0.add_input_port(0)
        node_1.add_input_port(0)
        node_2.add_input_port(0)

        self.assertEqual(node_0.in_port(0).get_source(), None)
        self.assertEqual(node_1.in_port(0).get_source(), None)
        self.assertEqual(node_2.in_port(0).get_source(), None)

    def test_port_disconnect_1(self):
        #              ,-->1-->1_data                   0-->0_data
        #   0-->0_data/--->2-->2_data  ==> 0-->0_data   1-->1_data
        #                                               2-->2_data
        graph = build_graph(self.nodes, [
            ('0', '0_data'),
            ('1', '1_data'),
            ('2', '2_data'),
            ('0_data', '1'),
            ('0_data', '2')
        ])
        graph.__setattr__('stage', 'middle')

        node_0 = Node(graph, '0')
        node_1 = Node(graph, '1')
        node_2 = Node(graph, '2')

        node_0.out_port(0).disconnect()

        self.assertEqual(node_1.in_port(0).get_source(), None)
        self.assertEqual(node_2.in_port(0).get_source(), None)

        self.assertTrue(len(node_1.in_nodes()) == 0)
        self.assertTrue(len(node_2.in_nodes()) == 0)

    def test_port_disconnect_2(self):
        #              ,-->1-->1_data                 ,-->1-->1_data
        #   0-->0_data/--->2-->2_data  ==> 0-->0_data/    2-->2_data
        #
        graph = build_graph(self.nodes, [
            ('0', '0_data'),
            ('1', '1_data'),
            ('2', '2_data'),
            ('0_data', '1'),
            ('0_data', '2')
        ])
        graph.__setattr__('stage', 'middle')

        node_0 = Node(graph, '0')
        node_1 = Node(graph, '1')
        node_2 = Node(graph, '2')

        node_2.in_port(0).disconnect()

        self.assertEqual(node_0.out_port(0).get_destination(), node_1.in_port(0))
        self.assertEqual(node_1.in_port(0).get_source(), node_0.out_port(0))
        self.assertEqual(node_2.out_port(0).get_destination(), None)
        self.assertEqual(node_2.in_port(0).get_source(), None)

        self.assertTrue(len(node_0.out_nodes()) == 1)
        self.assertTrue(len(node_1.in_nodes()) == 1)
        self.assertTrue(len(node_2.in_nodes()) == 0)

    def test_port_disconnect_3(self):
        #   1-->1_data---\                 1-->1_data
        #   0-->0_data---->2-->2_data  ==> 0-->0_data-->2-->2_data
        #
        graph = build_graph(self.nodes, [
            ('0', '0_data'),
            ('1', '1_data'),
            ('2', '2_data'),
            ('0_data', '2'),
            ('1_data', '2')
        ])
        graph.__setattr__('stage', 'middle')

        node_0 = Node(graph, '0')
        node_1 = Node(graph, '1')
        node_2 = Node(graph, '2')

        node_2.in_port(1).disconnect()

        self.assertEqual(node_0.out_port(0).get_destination(), node_2.in_port(0))
        self.assertEqual(node_2.in_port(0).get_source(), node_0.out_port(0))
        self.assertEqual(node_1.out_port(0).get_destination(), None)

        self.assertTrue(len(node_0.out_nodes()) == 1)
        self.assertTrue(len(node_1.in_nodes()) == 0)
        self.assertTrue(len(node_2.in_nodes()) == 1)

    def test_port_disconnect_4(self):
        #   1-->1_data---\                 0-->0_data
        #   0-->0_data---->2-->2_data  ==> 1-->1_data-->2-->2_data
        #
        graph = build_graph(self.nodes, [
            ('0', '0_data'),
            ('1', '1_data'),
            ('2', '2_data'),
            ('0_data', '2'),
            ('1_data', '2')
        ])
        graph.__setattr__('stage', 'middle')

        node_0 = Node(graph, '0')
        node_1 = Node(graph, '1')
        node_2 = Node(graph, '2')

        node_2.in_port(0).disconnect()

        self.assertEqual(node_1.out_port(0).get_destination(), node_2.in_port(1))
        self.assertEqual(node_2.in_port(1).get_source(), node_1.out_port(0))
        self.assertEqual(node_2.in_port(0).get_source(), None)
        self.assertEqual(node_0.out_port(0).get_destination(), None)
        #
        # self.assertTrue(len(node_0.out_nodes()) == 1)
        # self.assertTrue(len(node_1.in_nodes()) == 0)
        # self.assertTrue(len(node_2.in_nodes()) == 1)

      ###########################################
     ### TESTS FOR CONNECTION CLASS METHODS ####
    ###########################################

    def test_connection_set_source_1(self):
        graph = build_graph(self.nodes, [
            ('0', '0_data'),
            ('1', '1_data'),
            ('2', '2_data'),
            ('3', '3_data'),
            ('4', '4_data'),

            ('0_data', '1'),
            ('0_data', '2'),
            ('3_data', '4'),
        ])
        graph.__setattr__('stage', 'middle')

        node_0 = Node(graph, '0')
        node_1 = Node(graph, '1')
        node_2 = Node(graph, '2')
        node_3 = Node(graph, '3')
        node_4 = Node(graph, '4')

        c = node_0.out_port(0).get_connection()
        c.set_source(node_3.out_port(0))

        self.assertTrue(node_0.out_node().kind == 'data')

        self.assertEqual(node_0.out_port(0).get_destinations(), [])
        destinations = node_3.out_port(0).get_destinations()
        for port in destinations:
            self.assertTrue(port in [node_1.in_port(0), node_2.in_port(0), node_4.in_port(0)])

    def test_connection_set_source_2(self):
        # 2-->2_data                                  ,->2-->2_data
        # 0-->0_data-->1-->1_data   ==>    0-->0_data/-->1-->1_data
        #
        graph = build_graph(self.nodes, [
            ('0', '0_data'),
            ('1', '1_data'),
            ('2', '2_data'),

            ('0_data', '1'),
        ])
        graph.__setattr__('stage', 'middle')

        node_0 = Node(graph, '0')
        node_2 = Node(graph, '2')
        node_2.add_input_port(0)

        node_2.in_port(0).get_connection().set_source(node_0.out_port(0))

        graph_ref = build_graph(self.nodes, [
            ('0', '0_data'),
            ('1', '1_data'),
            ('2', '2_data'),

            ('0_data', '1'),
            ('0_data', '2'),
        ])

        (flag, resp) = compare_graphs(graph, graph_ref, '0', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_connection_set_source_3(self):
        #            ,->2-->2_data          0-->0_data-->1-->1_data
        # 0-->0_data/-->1-->1_data    =>    3-->3_data-->2-->2_data
        # 3-->3_data
        graph = build_graph(self.nodes, [
            ('0', '0_data'),
            ('1', '1_data'),
            ('2', '2_data'),
            ('3', '3_data'),

            ('0_data', '1'),
            ('0_data', '2'),
        ])
        graph.__setattr__('stage', 'middle')

        node_0 = Node(graph, '0')
        node_2 = Node(graph, '2')
        node_3 = Node(graph, '3')

        node_2.in_port(0).get_connection().set_source(node_3.out_port(0))

        graph_ref = build_graph(self.nodes, [
            ('0', '0_data'),
            ('1', '1_data'),
            ('2', '2_data'),
            ('3', '3_data'),

            ('0_data', '1'),
            ('3_data', '2'),
        ])

        (flag, resp) = compare_graphs(graph, graph_ref, '0', check_op_attrs=True)
        self.assertTrue(flag, resp)

        (flag, resp) = compare_graphs(graph, graph_ref, '2', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_connection_set_source_4(self):
        # 0   1   ==>    0-->1
        graph = build_graph(self.nodes, [])
        graph.__setattr__('stage', 'middle')

        node_0 = Node(graph, '0')
        node_1 = Node(graph, '1')

        node_0.add_output_port(0)
        node_1.add_input_port(0)

        node_1.in_port(0).get_connection().set_source(node_0.out_port(0))

        graph_ref = build_graph(self.nodes, [
            ('0', '0_data'),
            ('0_data', '1'),
        ])

        (flag, resp) = compare_graphs(graph, graph_ref, '0', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_connection_set_destination(self):
        #            ,->2-->2_data-->3-->3_data               ,->2-->2_data
        # 0-->0_data/-->1-->1_data   ==>           0-->0_data/-->3-->3_data
        #
        graph = build_graph(self.nodes, [
            ('0', '0_data'),
            ('1', '1_data'),
            ('2', '2_data'),
            ('3', '3_data'),

            ('0_data', '1'),
            ('0_data', '2'),
            ('2_data', '3'),
        ])
        graph.__setattr__('stage', 'middle')

        graph_ref = build_graph(self.nodes, [
            ('0', '0_data'),
            ('2', '2_data'),
            ('3', '3_data'),

            ('0_data', '3'),
            ('0_data', '2'),
        ])

        node_1 = Node(graph, '1')
        node_3 = Node(graph, '3')

        node_3.in_port(0).disconnect()
        node_1.in_port(0).get_connection().set_destination(node_3.in_port(0))

        (flag, resp) = compare_graphs(graph, graph_ref, '0', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_connection_set_destination_1(self):
        # 2
        #            ,->1-->1_data                    ,->2
        # 0-->0_data/-->1-->1_data   ==>   0-->0_data/-->1
        #
        graph = build_graph(self.nodes, [
            ('0', '0_data'),

            ('0_data', '1'),
            ('0_data', '1'),
        ])
        graph.__setattr__('stage', 'middle')

        graph_ref = build_graph(self.nodes, [
            ('0', '0_data'),

            ('0_data', '1'),
            ('0_data', '2'),
        ])

        node_1 = Node(graph, '1')
        node_2 = Node(graph, '2')
        node_2.add_input_port(0)

        node_1.in_port(1).get_connection().set_destination(node_2.in_port(0))

        (flag, resp) = compare_graphs(graph, graph_ref, '0', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_connection_set_destination_2(self):
        # 2
        #            ,->1                    ,->1
        # 0-->0_data/-->1   ==>   0-->0_data/-->2
        #
        graph = build_graph(self.nodes, [
            ('0', '0_data'),

            ('0_data', '1'),
            ('0_data', '1'),
        ])
        graph.__setattr__('stage', 'middle')

        graph_ref = build_graph(self.nodes, [
            ('0', '0_data'),

            ('0_data', '1', {'in': 1}),
            ('0_data', '2'),
        ])

        node_1 = Node(graph, '1')
        node_2 = Node(graph, '2')
        node_2.add_input_port(0)

        node_1.in_port(0).get_connection().set_destination(node_2.in_port(0))

        (flag, resp) = compare_graphs(graph, graph_ref, '0', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_connection_add_destination_1(self):
        # 3-->3_data                                     ,-->3-->3_data
        #            ,->2-->2_data                      ,-->2-->2_data
        # 0-->0_data/-->1-->1_data   ==>     0-->0_data/-->1-->1_data
        #
        graph = build_graph(self.nodes, [
            ('0', '0_data'),
            ('1', '1_data'),
            ('2', '2_data'),
            ('3', '3_data'),

            ('0_data', '1'),
            ('0_data', '2'),
        ])
        graph.__setattr__('stage', 'middle')

        graph_ref = build_graph(self.nodes, [
            ('0', '0_data'),
            ('1', '1_data'),
            ('2', '2_data'),
            ('3', '3_data'),

            ('0_data', '1'),
            ('0_data', '2'),
            ('0_data', '3'),
        ])

        node_0 = Node(graph, '0')
        node_3 = Node(graph, '3')
        node_3.add_input_port(idx=0)

        node_0.out_port(0).get_connection().add_destination(node_3.in_port(0))

        (flag, resp) = compare_graphs(graph, graph_ref, '0', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_connection_add_destination_2(self):
        # 0
        # 1-->1_data   ==>     0-->0_data-->1-->1_data
        graph = build_graph(self.nodes, [
            ('1', '1_data'),
        ])
        graph.__setattr__('stage', 'middle')

        graph_ref = build_graph(self.nodes, [
            ('0', '0_data'),
            ('1', '1_data'),
            ('0_data', '1'),
        ])

        node_0 = Node(graph, '0')
        node_0.add_output_port(idx=0)

        node_1 = Node(graph, '1')
        node_1.add_input_port(idx=0)

        node_0.out_port(0).get_connection().add_destination(node_1.in_port(0))

        (flag, resp) = compare_graphs(graph, graph_ref, '0', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_connection_get_source_destinations_1(self):
        graph = build_graph(self.nodes, [
            ('0', '0_data'),
            ('1', '1_data'),
            ('2', '2_data'),

            ('0_data', '1'),
            ('0_data', '2')
        ])
        graph.__setattr__('stage', 'middle')

        node_0 = Node(graph, '0')
        node_1 = Node(graph, '1')
        node_2 = Node(graph, '2')

        source = node_0.out_port(0).get_connection().get_source()
        destinations = node_0.out_port(0).get_connection().get_destinations()

        self.assertEqual(source, node_0.out_port(0))
        for port in destinations:
            self.assertTrue(port in [node_1.in_port(0), node_2.in_port(0)])

        self.assertEqual(node_1.out_port(0).get_connection().get_destination(), None)
        self.assertEqual(node_1.out_port(0).get_destination(), None)

        self.assertEqual(node_2.out_port(0).get_connection().get_destination(), None)
        self.assertEqual(node_2.out_port(0).get_destination(), None)

    def test_connection_remove_1(self):
        graph = build_graph(self.nodes, [
            ('0', '0_data'),
            ('1', '1_data'),
            ('2', '2_data'),

            ('0_data', '1'),
            ('0_data', '2')
        ])
        graph.__setattr__('stage', 'middle')

        node_0 = Node(graph, '0')
        node_1 = Node(graph, '1')
        node_2 = Node(graph, '2')

        node_1.in_port(0).get_connection().remove()

        self.assertEqual(node_0.out_port(0).get_destinations(), [node_2.in_port(0)])
        self.assertEqual(node_1.in_port(0).get_source(), None)
        self.assertEqual(node_2.in_port(0).get_source(), node_0.out_port(0))

    def test_connection_remove_2(self):
        graph = build_graph(self.nodes, [
            ('0', '0_data'),
            ('1', '1_data'),
            ('2', '2_data'),

            ('0_data', '1'),
            ('0_data', '2')
        ])
        graph.__setattr__('stage', 'middle')

        node_0 = Node(graph, '0')
        node_1 = Node(graph, '1')
        node_2 = Node(graph, '2')

        node_0.out_port(0).get_connection().remove()

        self.assertEqual(node_0.out_port(0).get_destinations(), [])
        self.assertEqual(node_1.out_port(0).get_destinations(), [])
        self.assertEqual(node_2.out_port(0).get_destinations(), [])

    def test_connection_data_1(self):
        graph = build_graph(self.nodes, [
            ('0', '0_data'),
            ('1', '1_data'),
            ('2', '2_data'),

            ('0_data', '1'),
            ('0_data', '2')
        ], {'0_data': {'value': np.ones((1,3,64,64)), 'shape': np.array([1, 3, 64, 64])}})

        graph.__setattr__('stage', 'middle')

        node_0 = Node(graph, '0')
        node_1 = Node(graph, '1')
        node_2 = Node(graph, '2')

        self.assertTrue(np.array_equal(node_0.out_port(0).get_connection().data.get_shape(), (1, 3, 64, 64)))
        self.assertTrue(np.array_equal(node_0.out_port(0).get_connection().data.get_value(), np.ones((1, 3, 64, 64))))

        self.assertEqual(node_1.out_port(0).get_connection().data.get_shape(), None)
        self.assertEqual(node_1.out_port(0).get_connection().data.get_value(), None)

        self.assertEqual(node_2.out_port(0).get_connection().data.get_shape(), None)
        self.assertEqual(node_2.out_port(0).get_connection().data.get_value(), None)

      ###########################################
     ################## OTHER ##################
    ###########################################

    def test_graph_cleanup_that_restores_const_operations(self):
        graph = build_graph(self.nodes, [
            ('0', '0_data'),
            ('1', '1_data'),
            ('2', '2_data'),
            ('3', '3_data'),

            ('0_data', '1'),
            ('2_data', '1'),
            ('3_data', '2'),
        ], {
            '3': {'shape': np.array([1, 227, 227, 3]), 'value': np.ones((1, 227, 227, 3))},
            '3_data': {'shape': np.array([1, 227, 227, 3]), 'value': np.ones((1, 227, 227, 3))},
            '2': {'shape': np.array([1, 227, 227, 3]), 'value': np.ones((1, 227, 227, 3))},
            '2_data': {'shape': np.array([1, 227, 227, 3]), 'value': np.ones((1, 227, 227, 3))},
        }, nodes_with_edges_only=True)
        add_opoutput(graph, '1_data', 0, False)

        graph_ref = build_graph(self.nodes, [
            ('0', '0_data'),
            ('1', '1_data'),
            ('const_1', '2_data'),

            ('0_data', '1'),
            ('2_data', '1'),
        ], {
            'const_1': {'shape': np.array([1, 227, 227, 3]), 'value': np.ones((1, 227, 227, 3))},
            '2_data': {'shape': np.array([1, 227, 227, 3]), 'value': np.ones((1, 227, 227, 3))},
        }, nodes_with_edges_only=True)
        add_opoutput(graph_ref, '1_data', 0, False)

        graph.clean_up()
        graph_ref.clean_up()

        (flag, resp) = compare_graphs(graph, graph_ref, '1_data', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_node_in_out_ports_order(self):
        #
        #               ,->2-->2_data---,->3-->3_data
        #   0-->0_data-/-->1-->1_data--/
        #
        graph = build_graph(self.nodes, [
            ('0', '0_data'),
            ('1', '1_data'),
            ('2', '2_data'),
            ('3', '3_data'),

            ('0_data', '1'),
            ('0_data', '2'),
            ('1_data', '3'),
            ('2_data', '3'),
        ])

        for id in ['0', '1', '2', '3']:
            node = Node(graph, id)
            for idx in range(len(node.in_ports())):
                self.assertEqual(node.in_port(idx), node.in_ports()[idx])
            for idx in range(len(node.out_ports())):
                self.assertEqual(node.out_port(idx), node.out_ports()[idx])


class TestNewGraphAPIFront(unittest.TestCase):
    nodes = {
        '0': {'type': 'Parameter', 'value': None, 'kind': 'op', 'op': 'Parameter'},
        '1': {'type': 'Parameter', 'value': None, 'kind': 'op', 'op': 'Parameter'},
        '2': {'type': 'Parameter', 'value': None, 'kind': 'op', 'op': 'Parameter'},
        '3': {'type': 'Parameter', 'value': None, 'kind': 'op', 'op': 'Parameter'},
        '4': {'type': 'Parameter', 'value': None, 'kind': 'op', 'op': 'Parameter'},
        'const_1': {'type': 'Const', 'value': None, 'kind': 'op', 'op': 'Const'},
    }

    ###########################################
    ###### TESTS FOR PORT CLASS METHODS #######
    ###########################################

    def test_port_get_destinations_1(self):
        #     ,->2
        #    /-->1
        #   0
        graph = build_graph(self.nodes, [
            ('0', '1', {'out': 0}),
            ('0', '2', {'out': 0}),
        ])
        graph.__setattr__('stage', 'front')

        node_0_out_port = Node(graph, '0').out_port(0)

        node_1_in_port = Node(graph, '1').in_port(0)
        node_2_in_port = Node(graph, '2').in_port(0)

        ports = node_0_out_port.get_destinations()

        self.assertTrue(len(ports) == 2)
        for port in ports:
            self.assertTrue(port in [node_1_in_port, node_2_in_port])

    def test_port_get_destination_1(self):
        #     ,->2
        #    /-->1
        #   0
        graph = build_graph(self.nodes, [
            ('0', '1', {'out': 0}),
            ('0', '2', {'out': 0}),
        ])
        graph.__setattr__('stage', 'front')

        node_0_out_port = Node(graph, '0').out_port(0)

        node_1_in_port = Node(graph, '1').in_port(0)
        node_2_in_port = Node(graph, '2').in_port(0)

        with self.assertRaises(Error):
            node_0_out_port.get_destination()

    def test_port_get_destination_2(self):
        graph = build_graph(self.nodes, [
            ('0', '1'),
        ])
        graph.__setattr__('stage', 'front')

        node_0_out_port = Node(graph, '0').out_port(0)

        node_1_in_port = Node(graph, '1').in_port(0)

        self.assertEqual(node_0_out_port.get_destination(), node_1_in_port)

    def test_port_get_destination_3(self):
        graph = build_graph(self.nodes, [
            ('0', '1', {'out': 0, 'in': 0}),
            ('0', '2', {'out': 1, 'in': 0}),
            ('0', '3', {'out': 1, 'in': 0}),
        ])
        graph.__setattr__('stage', 'front')

        node_0_out_port_1 = Node(graph, '0').out_port(1)
        node_2_in_port = Node(graph, '2').in_port(0)
        node_3_in_port = Node(graph, '3').in_port(0)

        destinations = node_0_out_port_1.get_destinations()

        self.assertTrue((destinations[0] == node_2_in_port and destinations[1] == node_3_in_port) or
                        (destinations[1] == node_2_in_port and destinations[0] == node_3_in_port))

    def test_port_get_source_1(self):
        graph = build_graph(self.nodes, [
            ('0', '1'),
        ])
        graph.__setattr__('stage', 'front')

        node_0_out_port = Node(graph, '0').out_port(0)

        node_1_in_port = Node(graph, '1').in_port(0)

        self.assertEqual(node_1_in_port.get_source(), node_0_out_port)

    def test_port_get_source_2(self):
        graph = build_graph(self.nodes, [
            ('0', '1'),
            ('2', '1')
        ])
        graph.__setattr__('stage', 'front')

        node_0 = Node(graph, '0')
        node_1 = Node(graph, '1')
        node_2 = Node(graph, '2')

        self.assertEqual(node_1.in_port(0).get_source(), node_0.out_port(0))
        self.assertEqual(node_1.in_port(1).get_source(), node_2.out_port(0))

    def test_port_get_source_3(self):
        graph = build_graph(self.nodes, [])
        graph.__setattr__('stage', 'front')

        node_0 = Node(graph, '0')
        node_1 = Node(graph, '1')
        node_2 = Node(graph, '2')

        node_0.add_input_port(0)
        node_1.add_input_port(0)
        node_2.add_input_port(0)

        self.assertEqual(node_0.in_port(0).get_source(), None)
        self.assertEqual(node_1.in_port(0).get_source(), None)
        self.assertEqual(node_2.in_port(0).get_source(), None)

    def test_port_disconnect_1(self):
        #              ,-->1-->1_data                   0-->0_data
        #   0-->0_data/--->2-->2_data  ==> 0-->0_data   1-->1_data
        #                                               2-->2_data
        graph = build_graph(self.nodes, [
            ('0', '1', {'out': 0}),
            ('0', '2', {'out': 0})
        ])
        graph.__setattr__('stage', 'front')

        node_0 = Node(graph, '0')
        node_1 = Node(graph, '1')
        node_2 = Node(graph, '2')

        node_0.out_port(0).disconnect()

        self.assertEqual(node_1.in_port(0).get_source(), None)
        self.assertEqual(node_2.in_port(0).get_source(), None)

        self.assertTrue(len(node_1.in_nodes()) == 0)
        self.assertTrue(len(node_2.in_nodes()) == 0)

    def test_port_disconnect_2(self):
        #        ,-->1           ,-->1
        #   0-->/--->2  ==> 0-->/    2
        #
        graph = build_graph(self.nodes, [
            ('0', '1', {'out': 0}),
            ('0', '2', {'out': 0})
        ])
        graph.__setattr__('stage', 'front')

        node_0 = Node(graph, '0')
        node_1 = Node(graph, '1')
        node_2 = Node(graph, '2')

        node_2.in_port(0).disconnect()

        self.assertEqual(node_0.out_port(0).get_destination(), node_1.in_port(0))
        self.assertEqual(node_1.in_port(0).get_source(), node_0.out_port(0))
        self.assertEqual(node_2.in_port(0).get_source(), None)

        self.assertTrue(len(node_0.out_nodes()) == 1)
        self.assertTrue(len(node_1.in_nodes()) == 1)
        self.assertTrue(len(node_2.in_nodes()) == 0)

    def test_port_disconnect_3(self):
        #   1---\          1
        #   0---->2    ==> 0-->2
        #
        graph = build_graph(self.nodes, [
            ('0', '2'),
            ('1', '2')
        ])
        graph.__setattr__('stage', 'front')

        node_0 = Node(graph, '0')
        node_1 = Node(graph, '1')
        node_2 = Node(graph, '2')

        node_2.in_port(1).disconnect()

        self.assertEqual(node_0.out_port(0).get_destination(), node_2.in_port(0))
        self.assertEqual(node_2.in_port(0).get_source(), node_0.out_port(0))
        self.assertEqual(node_1.out_port(0).get_destination(), None)

        self.assertTrue(len(node_0.out_nodes()) == 1)
        self.assertTrue(len(node_1.in_nodes()) == 0)
        self.assertTrue(len(node_2.in_nodes()) == 1)

    def test_port_disconnect_4(self):
        #   1-----\        0
        #   0------>2  ==> 1--->2
        #
        graph = build_graph(self.nodes, [
            ('0', '2', {'out': 0}),
            ('1', '2', {'out': 0})
        ])
        graph.__setattr__('stage', 'front')

        node_0 = Node(graph, '0')
        node_1 = Node(graph, '1')
        node_2 = Node(graph, '2')

        node_2.in_port(0).disconnect()

        self.assertEqual(node_1.out_port(0).get_destination(), node_2.in_port(1))
        self.assertEqual(node_2.in_port(1).get_source(), node_1.out_port(0))
        self.assertEqual(node_2.in_port(0).get_source(), None)
        self.assertEqual(node_0.out_port(0).get_destination(), None)

    def test_port_disconnected_1(self):
        graph = build_graph(self.nodes, [
            ('0', '1'),
            ('1', '2')
        ])
        graph.__setattr__('stage', 'front')

        node_0 = Node(graph, '0')
        node_1 = Node(graph, '1')
        node_2 = Node(graph, '2')
        node_2.add_output_port(0)
        node_0.add_input_port(0)

        self.assertTrue(not node_0.out_port(0).disconnected())
        self.assertTrue(not node_1.out_port(0).disconnected())
        self.assertTrue(not node_1.in_port(0).disconnected())
        self.assertTrue(node_2.out_port(0).disconnected())
        self.assertTrue(node_0.in_port(0).disconnected())

    def test_port_get_connection_1(self):
        graph = build_graph(self.nodes, [
            ('0', '1'),
            ('1', '2', {'out': 0}),
            ('1', '3', {'out': 0}),
        ])
        graph.__setattr__('stage', 'front')

        node_1 = Node(graph, '1')
        node_2 = Node(graph, '3')
        node_3 = Node(graph, '2')

        c = node_1.out_port(0).get_connection()

        self.assertTrue(c.get_source() == node_1.out_port(0))
        for port in c.get_destinations():
            self.assertTrue(port in [node_2.in_port(0), node_3.in_port(0)])

    ###########################################
    ### TESTS FOR CONNECTION CLASS METHODS ####
    ###########################################

    def test_connection_set_source_1(self):
        graph = build_graph(self.nodes, [
            ('0', '1'),
            ('0', '2'),
            ('3', '4'),
        ])
        graph.__setattr__('stage', 'front')

        node_0 = Node(graph, '0')
        node_1 = Node(graph, '1')
        node_2 = Node(graph, '2')
        node_3 = Node(graph, '3')
        node_4 = Node(graph, '4')

        c = node_0.out_port(0).get_connection()
        c.set_source(node_3.out_port(0))

        self.assertEqual(node_0.out_port(0).get_destinations(), [])
        destinations = node_3.out_port(0).get_destinations()
        for port in destinations:
            self.assertTrue(port in [node_1.in_port(0), node_2.in_port(0), node_4.in_port(0)])

    def test_connection_set_source_2(self):
        # 2                ,->2
        # 0-->1   ==>    0/-->1
        #
        graph = build_graph(self.nodes, [
            ('0', '1'),
        ])
        graph.__setattr__('stage', 'front')

        node_0 = Node(graph, '0')
        node_2 = Node(graph, '2')
        node_2.add_input_port(0)

        node_2.in_port(0).get_connection().set_source(node_0.out_port(0))

        graph_ref = build_graph(self.nodes, [
            ('0', '1', {'out': 0, 'in': 0}),
            ('0', '2', {'out': 0, 'in': 0}),
        ])

        (flag, resp) = compare_graphs(graph, graph_ref, '0', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_connection_set_source_3(self):
        # 0   1   ==>    0-->1
        graph = build_graph(self.nodes, [])
        graph.__setattr__('stage', 'front')

        node_0 = Node(graph, '0')
        node_1 = Node(graph, '1')

        node_0.add_output_port(0)
        node_1.add_input_port(0)

        node_1.in_port(0).get_connection().set_source(node_0.out_port(0))

        graph_ref = build_graph(self.nodes, [
            ('0', '1', {'out': 0, 'in': 0}),
        ])

        (flag, resp) = compare_graphs(graph, graph_ref, '0', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_connection_set_destination(self):
        #            ,->2-->2_data-->3-->3_data               ,->2-->2_data
        # 0-->0_data/-->1-->1_data   ==>           0-->0_data/-->3-->3_data
        #
        graph = build_graph(self.nodes, [
            ('0', '1'),
            ('0', '2'),
            ('2', '3'),
        ])
        graph.__setattr__('stage', 'front')

        graph_ref = build_graph(self.nodes, [
            ('0', '3'),
            ('0', '2'),
        ])

        node_1 = Node(graph, '1')
        node_3 = Node(graph, '3')

        node_3.in_port(0).disconnect()
        node_1.in_port(0).get_connection().set_destination(node_3.in_port(0))

        (flag, resp) = compare_graphs(graph, graph_ref, '0', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_connection_set_destination_1(self):
        # 2
        #  ,->1          ,->2
        # 0-->1   ==>   0-->1
        #
        graph = build_graph(self.nodes, [
            ('0', '1', {'out': 0, 'in': 0}),
            ('0', '1', {'out': 0, 'in': 1}),
        ])
        graph.__setattr__('stage', 'front')

        graph_ref = build_graph(self.nodes, [
            ('0', '1', {'out': 0, 'in': 0}),
            ('0', '2', {'out': 0, 'in': 0}),
        ])

        node_1 = Node(graph, '1')
        node_2 = Node(graph, '2')
        node_2.add_input_port(0)

        node_1.in_port(1).get_connection().set_destination(node_2.in_port(0))

        (flag, resp) = compare_graphs(graph, graph_ref, '0', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_connection_set_destination_2(self):
        # 2
        #  ,->1          ,->1
        # 0-->1   ==>   0-->2
        #
        graph = build_graph(self.nodes, [
            ('0', '1', {'out': 0, 'in': 0}),
            ('0', '1', {'out': 0, 'in': 1}),
        ])
        graph.__setattr__('stage', 'front')

        graph_ref = build_graph(self.nodes, [
            ('0', '1', {'out': 0, 'in': 1}),
            ('0', '2', {'out': 0, 'in': 0}),
        ])

        node_1 = Node(graph, '1')
        node_2 = Node(graph, '2')
        node_2.add_input_port(0)

        node_1.in_port(0).get_connection().set_destination(node_2.in_port(0))

        (flag, resp) = compare_graphs(graph, graph_ref, '0', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_connection_add_destination_1(self):
        # 3                   ,-->3
        #     ,->2           ,-->2
        # 0--/-->1  ==>  0--/-->1
        #
        graph = build_graph(self.nodes, [
            ('0', '1', {'in': 0, 'out': 0}),
            ('0', '2', {'in': 0, 'out': 0}),
        ])
        graph.__setattr__('stage', 'front')

        graph_ref = build_graph(self.nodes, [
            ('0', '1', {'in': 0, 'out': 0}),
            ('0', '2', {'in': 0, 'out': 0}),
            ('0', '3', {'in': 0, 'out': 0}),
        ])

        node_0 = Node(graph, '0')
        node_3 = Node(graph, '3')
        node_3.add_input_port(idx=0)

        node_0.out_port(0).get_connection().add_destination(node_3.in_port(0))

        (flag, resp) = compare_graphs(graph, graph_ref, '0', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_connection_add_destination_2(self):
        # 0
        # 1   ==>     0-->1
        graph = build_graph(self.nodes, [])
        graph.__setattr__('stage', 'front')

        graph_ref = build_graph(self.nodes, [
            ('0', '1'),
        ])

        node_0 = Node(graph, '0')
        node_0.add_output_port(idx=0)

        node_1 = Node(graph, '1')
        node_1.add_input_port(idx=0)

        node_0.out_port(0).get_connection().add_destination(node_1.in_port(0))

        (flag, resp) = compare_graphs(graph, graph_ref, '0', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_connection_get_source_destinations_1(self):
        graph = build_graph(self.nodes, [
            ('0', '1'),
            ('0', '2')
        ])
        graph.__setattr__('stage', 'front')

        node_0 = Node(graph, '0')
        node_1 = Node(graph, '1')
        node_2 = Node(graph, '2')
        node_1.add_output_port(idx=0)
        node_2.add_output_port(idx=0)

        source = node_0.out_port(0).get_connection().get_source()
        destinations = node_0.out_port(0).get_connection().get_destinations()

        self.assertEqual(source, node_0.out_port(0))
        for port in destinations:
            self.assertTrue(port in [node_1.in_port(0), node_2.in_port(0)])

        self.assertEqual(node_1.out_port(0).get_connection().get_destination(), None)
        self.assertEqual(node_1.out_port(0).get_destination(), None)

        self.assertEqual(node_2.out_port(0).get_connection().get_destination(), None)
        self.assertEqual(node_2.out_port(0).get_destination(), None)

    def test_connection_remove_1(self):
        graph = build_graph(self.nodes, [
            ('0', '1', {'in': 0, 'out': 0}),
            ('0', '2', {'in': 0, 'out': 0})
        ])
        graph.__setattr__('stage', 'front')

        node_0 = Node(graph, '0')
        node_1 = Node(graph, '1')
        node_2 = Node(graph, '2')

        node_1.in_port(0).get_connection().remove()

        self.assertEqual(node_0.out_port(0).get_destinations(), [node_2.in_port(0)])
        self.assertEqual(node_1.in_port(0).get_source(), None)
        self.assertEqual(node_2.in_port(0).get_source(), node_0.out_port(0))

    def test_connection_remove_2(self):
        graph = build_graph(self.nodes, [
            ('0', '1', {'in': 0, 'out': 0}),
            ('0', '2', {'in': 0, 'out': 0})
        ])
        graph.__setattr__('stage', 'front')

        node_0 = Node(graph, '0')
        node_1 = Node(graph, '1')
        node_2 = Node(graph, '2')

        node_0.out_port(0).get_connection().remove()

        self.assertEqual(node_0.out_port(0).get_destinations(), [])
        self.assertEqual(node_1.in_port(0).get_source(), None)
        self.assertEqual(node_2.in_port(0).get_source(), None)


class TestDictIncludesCompareAttrs(unittest.TestCase):
    def test_numpy_scalar(self):
        self.assertTrue(dict_includes_compare_attrs(2.0, np.array(2.0)))
        self.assertTrue(dict_includes_compare_attrs(2, np.array(2.0)))
        self.assertTrue(dict_includes_compare_attrs(np.array(2.0), 2.0))
        self.assertTrue(dict_includes_compare_attrs(np.array(2.0), 2))

        self.assertFalse(dict_includes_compare_attrs(2.01, np.array(2.0)))
        self.assertFalse(dict_includes_compare_attrs(2, np.array(2.1)))
        self.assertFalse(dict_includes_compare_attrs(np.array(2.0), 2.01))
        self.assertFalse(dict_includes_compare_attrs(np.array(2.1), 2))

    def test_regular_scalars(self):
        self.assertTrue(dict_includes_compare_attrs(2.0, 2))
        self.assertFalse(dict_includes_compare_attrs(2, 1.99999999999999))

    def test_lists_numpy(self):
        self.assertTrue(dict_includes_compare_attrs([4, 2, 3], np.array([4, 2, 3])))
        self.assertFalse(dict_includes_compare_attrs([4, 2, 3], np.array([1, 2, 3])))

    def test_regular_lists(self):
        self.assertTrue(dict_includes_compare_attrs([4, 2, 3], [4, 2, 3]))
        self.assertFalse(dict_includes_compare_attrs([4, 2, 3], [1, 2, 3]))
        self.assertFalse(dict_includes_compare_attrs([4, 2, 3], [4, 2, 3, 5]))

    def test_regular_string(self):
        self.assertTrue(dict_includes_compare_attrs("abc", "abc"))
        self.assertFalse(dict_includes_compare_attrs("abc", "abd"))


class TestGetSetAttributeBetweenNodes(unittest.TestCase):
    nodes = {
        'A': {'id': 0, 'kind': 'op'},
        'B': {'id': 1, 'kind': 'op'},
        'C': {'id': 2, 'kind': 'op'},
        'D': {'id': 3, 'kind': 'op'},
        'E': {'id': 4, 'kind': 'op'},
        'F': {'id': 5, 'kind': 'op'},
    }

    def build_test_graph(self):
        graph = build_graph(self.nodes, [
            ('A', 'D', {'in': 0, 'out': 0, 'Attr': "A-D"}),
            ('A', 'E', {'in': 0, 'out': 1, 'Attr': "A-E"}),
            ('A', 'F', {'in': 0, 'out': 2, 'Attr': "A-F"}),
            ('B', 'D', {'in': 1, 'out': 0, 'Attr': "B-D"}),
            ('B', 'F', {'in': 2, 'out': 1, 'Attr': "B-F"}),
        ])
        return graph

    def test_get_attribute_between_nodes(self):
        graph = self.build_test_graph()
        a_node = Node(graph, 'A')
        b_node = Node(graph, 'B')
        d_node = Node(graph, 'D')
        e_node = Node(graph, 'E')
        f_node = Node(graph, 'F')
        self.assertTrue(get_edge_attribute_between_nodes(a_node, d_node, 'Attr') == "A-D")
        self.assertTrue(get_edge_attribute_between_nodes(a_node, e_node, 'Attr') == "A-E")
        self.assertTrue(get_edge_attribute_between_nodes(a_node, f_node, 'Attr') == "A-F")
        self.assertTrue(get_edge_attribute_between_nodes(b_node, d_node, 'Attr') == "B-D")
        self.assertTrue(get_edge_attribute_between_nodes(b_node, f_node, 'Attr') == "B-F")

    def test_set_attribute_between_nodes(self):
        graph = self.build_test_graph()
        a_node = Node(graph, 'A')
        b_node = Node(graph, 'B')
        d_node = Node(graph, 'D')
        e_node = Node(graph, 'E')
        f_node = Node(graph, 'F')

        set_edge_attribute_between_nodes(a_node, d_node, 'Attr', 'new_value_1')
        set_edge_attribute_between_nodes(a_node, e_node, 'Attr', 'new_value_2')
        set_edge_attribute_between_nodes(a_node, f_node, 'Attr', 'new_value_3')
        set_edge_attribute_between_nodes(b_node, d_node, 'Attr', 'new_value_4')
        set_edge_attribute_between_nodes(b_node, f_node, 'Attr', 'new_value_5')

        self.assertTrue(get_edge_attribute_between_nodes(a_node, d_node, 'Attr') == "new_value_1")
        self.assertTrue(get_edge_attribute_between_nodes(a_node, e_node, 'Attr') == "new_value_2")
        self.assertTrue(get_edge_attribute_between_nodes(a_node, f_node, 'Attr') == "new_value_3")
        self.assertTrue(get_edge_attribute_between_nodes(b_node, d_node, 'Attr') == "new_value_4")
        self.assertTrue(get_edge_attribute_between_nodes(b_node, f_node, 'Attr') == "new_value_5")
