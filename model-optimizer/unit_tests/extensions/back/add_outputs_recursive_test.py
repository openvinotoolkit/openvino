# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import unittest

from extensions.back.add_outputs_recursive import AddOutputRecursive
from extensions.ops.loop import Loop
from extensions.ops.tensor_iterator import TensorIterator
from mo.front.common.partial_infer.elemental import copy_shape_infer
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node
from unit_tests.utils.graph import build_graph, regular_op_with_empty_data, result, connect, shaped_parameter, \
    valued_const_with_data, shaped_const_with_data, regular_op_with_shaped_data

# test for Loop
main_graph_nodes = {
    **shaped_parameter("IN_1", [1, 4, 64, 54]),
    **shaped_parameter("IN_2", [1, 4, 64, 54]),
    **valued_const_with_data("M", int64_array([5])),
    **valued_const_with_data("cond", int64_array([1])),
    **regular_op_with_empty_data("Loop", {'op': "Loop", 'type': 'Loop', "body": None,
                                          'input_port_map': [{'external_port_id': 1, 'internal_layer_id': 2, 'axis': None},
                                                             {'external_port_id': 2, 'internal_layer_id': 0, 'axis': None},
                                                             {'external_port_id': 3, 'internal_layer_id': 1, 'axis': None}],
                                          'output_port_map': [{'external_port_id': 4, 'internal_layer_id': 4,
                                                                'axis': None},
                                                              {'external_port_id': -1, 'internal_layer_id': 5,
                                                               'axis': None, 'purpose': "execution_condition"}],
                                          'back_edges': [{'from_layer': 8, 'to_layer': 7},
                                                         {'from_layer': 10, 'to_layer': 9}],
                                 'infer': Loop.infer
}),
    **result("OUT_1")
}

sub_graph_1_nodes = {
    **shaped_parameter("IN_2", int64_array([1, 4, 64, 54]), {'internal_layer_id': 0}),
    **valued_const_with_data("M_2", int64_array([10])),
    **valued_const_with_data("cond_2", int64_array([1])),
    **regular_op_with_empty_data("Loop_2", {'op': "Loop", 'type': 'Loop', "body": None,
                                            'input_port_map': [{'external_port_id': 1, 'internal_layer_id': 0, 'axis': None},
                                                               {'external_port_id': 2, 'internal_layer_id': 2, 'axis': None}],
                                            'output_port_map': [{'external_port_id': 3, 'internal_layer_id': 7,
                                                                 'axis': None},
                                                                {'external_port_id': -1, 'internal_layer_id': 6,
                                                                 'axis': None,
                                                                'purpose': "execution_condition"}],
                                            'back_edges': [{'from_layer': 1, 'to_layer': 0},
                                                           {'from_layer': 8, 'to_layer': 2}],
                                            'infer': Loop.infer}),
    **regular_op_with_empty_data('Loop_2_out', {'op': 'Result', 'type': 'Result', 'infer': lambda x: None,
                                                'internal_layer_id': 3}),
    **shaped_parameter("in_1_int", int64_array([1, 4, 64, 54]), {'internal_layer_id': 1}),
    **regular_op_with_empty_data("in_1_int_out", {'op': 'Result', 'type': 'Result', 'infer': lambda x: None,
                                                  'internal_layer_id': 4}),
    **shaped_parameter("cond_1_int", int64_array([1]), {'internal_layer_id': 2}),
    **regular_op_with_empty_data("cond_1_int_out", {'op': 'Result', 'type': 'Result', 'infer': lambda x: None,
                                                    'internal_layer_id': 5}),
}

sub_graph_2_nodes = {
    **shaped_parameter('cond_2_int', [1, 4, 64, 54], {'internal_layer_id': 0}),
    **result("cond_2_int_out"),
    **shaped_parameter('in_2_int', [1, 4, 64, 54], {'internal_layer_id': 1}),
    **shaped_const_with_data('ones', int64_array([1, 4, 64, 54])),
    **regular_op_with_empty_data('OUT_2', {'op': "Add", 'infer': copy_shape_infer}),
    **regular_op_with_empty_data('OUT_2_out', {'op': 'Result', 'type': 'Result', 'infer': lambda x: None,
                                               'internal_layer_id': 7}),
    **regular_op_with_empty_data('in_2_int_out', {'op': 'Result', 'type': 'Result', 'infer': lambda x: None,
                                                  'internal_layer_id': 6})
}


class AddOutputRecursiveTest(unittest.TestCase):

    def test_add_output_1(self):
        sub_graph_2 = build_graph(nodes_attrs=sub_graph_2_nodes,
                                  edges=[*connect('cond_2_int', 'cond_2_int_out'),
                                         *connect('in_2_int', 'OUT_2'),
                                         *connect('ones', 'OUT_2'),
                                         *connect('OUT_2', 'OUT_2_out'),
                                         *connect('in_2_int', 'in_2_int_out')],
                                  nodes_with_edges_only=True)

        sub_graph_1 = build_graph(nodes_attrs=sub_graph_1_nodes,
                                  edges=[*connect('M_2', '0:Loop_2'),
                                         *connect('cond_2', '1:Loop_2'),
                                         *connect('IN_2', '2:Loop_2'),
                                         *connect('Loop_2:3', 'Loop_2_out'),
                                         *connect('in_1_int', 'in_1_int_out'),
                                         *connect('cond_1_int', 'cond_1_int_out')],
                                  nodes_with_edges_only=True)
        loop_node_1 = Node(sub_graph_1, 'Loop_2')
        loop_node_1.body = sub_graph_2

        main_graph = build_graph(nodes_attrs=main_graph_nodes,
                                 edges=[*connect('M', '0:Loop'),
                                        *connect('cond', '1:Loop'),
                                        *connect('IN_2', '2:Loop'),
                                        *connect('IN_1', "3:Loop"),
                                        *connect('Loop:4', 'OUT_1')],
                                 nodes_with_edges_only=True)
        loop_node = Node(main_graph, 'Loop')
        loop_node.body = sub_graph_1
        main_graph.graph['additional_outputs'] = ['Loop', 'Loop_2']

        AddOutputRecursive().find_and_replace_pattern(main_graph)
        loop_node = Node(main_graph, 'Loop')
        self.assertEqual(len(loop_node.output_port_map), 3)
        self.assertEqual(len(loop_node.out_ports()), 2)
        self.assertEqual(loop_node.out_port(5).get_destination().node.op, 'Result')
        self.assertTrue(np.all(loop_node.out_port(5).data.get_shape() == int64_array([5, 1, 4, 64, 54])))
        last_node = Node(sub_graph_1, 'Loop_2')
        self.assertEqual(len(last_node.out_ports()), 1)
        unsq_node = last_node.out_port(3).get_destinations()[1].node
        self.assertEqual(unsq_node.op, 'Unsqueeze')
        self.assertEqual(unsq_node.out_port(0).get_destination().node.op, 'Result')
        self.assertTrue(np.all(unsq_node.out_port(0).data.get_shape() == int64_array([1, 1, 4, 64, 54])))

# test for TensorIterator
ti_main_graph_nodes = {
    **shaped_parameter("IN_1", [1, 4, 64, 54]),
    **shaped_parameter("IN_2", [1, 4, 64, 54]),
    **valued_const_with_data("M", int64_array([5])),
    **valued_const_with_data("cond", int64_array([1])),
    **regular_op_with_empty_data("Loop", {'op': "TensorIterator", 'type': 'TensorIterator',
                                          'sub_graphs': ['body'], "body": None,
                                          'input_port_map': [{'external_port_id': 1, 'internal_layer_id': 2, 'axis': None},
                                                             {'external_port_id': 2, 'internal_layer_id': 0, 'axis': None},
                                                             {'external_port_id': 3, 'internal_layer_id': 1, 'axis': None}],
                                          'output_port_map': [{'external_port_id': 4, 'internal_layer_id': 4,
                                                                'axis': None},
                                                              {'external_port_id': -1, 'internal_layer_id': 5,
                                                               'axis': None, 'purpose': "execution_condition"}],
                                          'back_edges': [{'from_layer': 8, 'to_layer': 7},
                                                         {'from_layer': 10, 'to_layer': 9}],
                                 'infer': TensorIterator.infer
}),
    **result("OUT_1")
}

ti_sub_graph_1_nodes = {
    **shaped_parameter("IN_2", int64_array([1, 4, 64, 54]), {'internal_layer_id': 0}),
    **valued_const_with_data("cond_2", int64_array([1])),
    **regular_op_with_empty_data("Loop_2", {'op': "TensorIterator", 'type': 'TensorIterator',
                                            'sub_graphs': ['body'], "body": None,
                                            'input_port_map': [{'external_port_id': 1, 'internal_layer_id': 0, 'axis': None},
                                                               {'external_port_id': 0, 'internal_layer_id': 1, 'axis': 0}],
                                            'output_port_map': [{'external_port_id': 2, 'internal_layer_id': 7,
                                                                 'axis': None},
                                                                ],
                                            'back_edges': [{'from_layer': 1, 'to_layer': 0},
                                                           {'from_layer': 8, 'to_layer': 2}],
                                            'infer': TensorIterator.infer}),
    **regular_op_with_empty_data('Loop_2_out', {'op': 'Result', 'type': 'Result', 'infer': lambda x: None,
                                                'internal_layer_id': 3}),
    **shaped_parameter("in_1_int", int64_array([1, 4, 64, 54]), {'internal_layer_id': 1}),
    **regular_op_with_empty_data("in_1_int_out", {'op': 'Result', 'type': 'Result', 'infer': lambda x: None,
                                                  'internal_layer_id': 4}),
    **shaped_parameter("cond_1_int", int64_array([1]), {'internal_layer_id': 2}),
    **regular_op_with_empty_data("cond_1_int_out", {'op': 'Result', 'type': 'Result', 'infer': lambda x: None,
                                                    'internal_layer_id': 5}),
}

ti_sub_graph_2_nodes = {
    **shaped_parameter('cond_2_int', [1, 4, 64, 54], {'internal_layer_id': 0}),
    **result("cond_2_int_out"),
    **shaped_parameter('in_2_int', [1, 4, 64, 54], {'internal_layer_id': 1}),
    **shaped_const_with_data('ones', int64_array([1, 4, 64, 54])),
    **regular_op_with_shaped_data('OUT_2', int64_array([1, 4, 64, 54]),
                                  {'op': "Add", 'infer': copy_shape_infer}),
    **regular_op_with_empty_data('OUT_2_out', {'op': 'Result', 'type': 'Result', 'infer': lambda x: None,
                                               'internal_layer_id': 7}),
    **regular_op_with_empty_data('in_2_int_out', {'op': 'Result', 'type': 'Result', 'infer': lambda x: None,
                                                  'internal_layer_id': 6})
}


class TI_AddOutputRecursiveTest(unittest.TestCase):

    def test_add_output_1(self):
        sub_graph_2 = build_graph(nodes_attrs=ti_sub_graph_2_nodes,
                                  edges=[*connect('cond_2_int', 'cond_2_int_out'),
                                         *connect('in_2_int', 'OUT_2'),
                                         *connect('ones', 'OUT_2'),
                                         *connect('OUT_2', 'OUT_2_out'),
                                         *connect('in_2_int', 'in_2_int_out')],
                                  nodes_with_edges_only=True)

        sub_graph_1 = build_graph(nodes_attrs=ti_sub_graph_1_nodes,
                                  edges=[*connect('cond_2', '1:Loop_2'),
                                         *connect('IN_2', '0:Loop_2'),
                                         *connect('Loop_2:2', 'Loop_2_out'),
                                         *connect('in_1_int', 'in_1_int_out'),
                                         *connect('cond_1_int', 'cond_1_int_out')],
                                  nodes_with_edges_only=True)
        loop_node_1 = Node(sub_graph_1, 'Loop_2')
        loop_node_1.body = sub_graph_2
        loop_node_1.in_edge(0)['external_port_id'] = 0
        loop_node_1.in_edge(1)['external_port_id'] = 1
        loop_node_1.out_edge(2)['external_port_id'] = 2

        main_graph = build_graph(nodes_attrs=ti_main_graph_nodes,
                                 edges=[*connect('M', '0:Loop'),
                                        *connect('cond', '1:Loop'),
                                        *connect('IN_2', '2:Loop'),
                                        *connect('IN_1', "3:Loop"),
                                        *connect('Loop:4', 'OUT_1')],
                                 nodes_with_edges_only=True)
        loop_node = Node(main_graph, 'Loop')
        loop_node.body = sub_graph_1
        loop_node.in_edge(0)['external_port_id'] = 0
        loop_node.in_edge(1)['external_port_id'] = 1
        loop_node.in_edge(2)['external_port_id'] = 2
        loop_node.in_edge(3)['external_port_id'] = 3
        loop_node.out_edge(4)['external_port_id'] = 4

        main_graph.graph['additional_outputs'] = ['Loop', 'Loop_2']

        AddOutputRecursive().find_and_replace_pattern(main_graph)
        loop_node = Node(main_graph, 'Loop')
        self.assertEqual(len(loop_node.output_port_map), 3)
        self.assertEqual(len(loop_node.out_ports()), 2)
        self.assertEqual(loop_node.out_port(5).get_destination().node.op, 'Result')
        self.assertTrue(np.all(loop_node.out_port(5).data.get_shape() == int64_array([1, 1, 4, 64, 54])))
        last_node = Node(sub_graph_1, 'Loop_2')
        self.assertEqual(len(last_node.out_ports()), 1)
        unsq_node = last_node.out_port(2).get_destinations()[1].node
        self.assertEqual(unsq_node.op, 'Unsqueeze')
        self.assertEqual(unsq_node.out_port(0).get_destination().node.op, 'Result')
        self.assertTrue(np.all(unsq_node.out_port(0).data.get_shape() == int64_array([1, 1, 4, 64, 54])))



