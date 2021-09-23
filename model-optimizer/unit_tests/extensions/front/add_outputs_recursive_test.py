# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from extensions.front.add_outputs_recursive import AddOutputRecursive
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node
from unit_tests.utils.graph import build_graph, regular_op, const, result, connect, shaped_parameter

main_graph_nodes = {
    **regular_op("IN_1", {'op': 'Parameter'}),
    **regular_op("IN_2", {'op': 'Parameter'}),
    **const("M", int64_array([0])),
    **const("cond", int64_array([0])),
    **regular_op("Loop", {'op': "Loop", "body": None,
                 'input_port_map': [{'external_port_id': 1, 'internal_layer_id': 7},
                                    {'external_port_id': 2, 'internal_layer_id': 9},
                                    {'external_port_id': 3, 'internal_layer_id': 0}],
                 'output_port_map': [{'external_port_id': -1, 'internal_layer_id': 8, 'purpose': "execution_condition"}],
                 'back_edges': [{'from_layer': 8, 'to_layer': 7},
                                {'from_layer': 10, 'to_layer': 9}]}),
    **result("OUT_1")
}

sub_graph_1_nodes = {
    **shaped_parameter("IN_2", int64_array([1, 4, 64, 54])),
    **const("M_2", int64_array([0])),
    **const("cond_2", int64_array([0])),
    **regular_op("Loop_2", {'op': "Loop", "body": None,
                            'input_port_map': [{'external_port_id': 1, 'internal_layer_id': 0},
                                               {'external_port_id': 2, 'internal_layer_id': 2}],
                            'output_port_map': [{'external_port_id': -1, 'internal_layer_id': 1,
                                                 'purpose': "execution_condition"}],
                            'back_edges': [{'from_layer': 1, 'to_layer': 0},
                                           {'from_layer': 8, 'to_layer': 2}]}),
    **const("Unsqueeze_const", int64_array([0])),
    **regular_op("Unsqueeze", {'op': "Unsqueeze", 'out_ports_count': 1}),
    **shaped_parameter("in_1_int", int64_array([1, 4, 64, 54])),
    **result("in_1_int_out"),
    **shaped_parameter("cond_1_int", int64_array([1])),
    **result("cond_1_int_out"),
}

sub_graph_2_nodes = {
    **regular_op('cond_2_int', {'op': 'Parameter'}),
    **result("cond_2_int_out"),
    **regular_op('in_2_int', {'op': 'Parameter'}),
    **const('ones', np.ones([1, 4, 64, 54])),
    **regular_op('OUT_2', {'op': "Add"}),
    **const("Unsqueeze_const_2", int64_array([0])),
    **regular_op("Unsqueeze_2", {'op': 'Unsqueeze'}),
    **result('in_2_int_out')
}


class AddOutputRecursiveTest(unittest.TestCase):

    def test_add_output_1(self):
        sub_graph_2 = build_graph(nodes_attrs=sub_graph_2_nodes,
                                  edges=[*connect('cond_2_int', 'cond_2_int_out', front_phase=True),
                                         *connect('in_2_int', 'OUT_2', front_phase=True),
                                         *connect('ones', 'OUT_2', front_phase=True),
                                         *connect('OUT_2', 'Unsqueeze_2', front_phase=True),
                                         *connect('Unsqueeze_const_2', 'Unsqueeze_2', front_phase=True),
                                         *connect('in_2_int', 'in_2_int_out', front_phase=True)],
                                  nodes_with_edges_only=True)
        sub_graph_2.stage = 'front'

        sub_graph_1 = build_graph(nodes_attrs=sub_graph_1_nodes,
                                  edges=[*connect('M_2', 'Loop_2', front_phase=True),
                                         *connect('cond_2', 'Loop_2', front_phase=True),
                                         *connect('IN_2', 'Loop_2', front_phase=True),
                                         *connect('Loop_2', 'Unsqueeze', front_phase=True),
                                         *connect('Unsqueeze_const', 'Unsqueeze', front_phase=True),
                                         *connect('in_1_int', 'in_1_int_out', front_phase=True),
                                         *connect('cond_1_int', 'cond_1_int_out', front_phase=True)],
                                  nodes_with_edges_only=True)
        loop_node_1 = Node(sub_graph_1, 'Loop_2')
        loop_node_1.body = sub_graph_2
        sub_graph_1.stage = 'front'

        main_graph = build_graph(nodes_attrs=main_graph_nodes,
                                 edges=[*connect('M', 'Loop', front_phase=True),
                                        *connect('cond', 'Loop', front_phase=True),
                                        *connect('IN_2', 'Loop', front_phase=True),
                                        *connect('IN_1', "Loop", front_phase=True),
                                        *connect('Loop', 'OUT_1', front_phase=True)],
                                 nodes_with_edges_only=True)
        loop_node = Node(main_graph, 'Loop')
        loop_node.body = sub_graph_1
        main_graph.stage = 'front'
        main_graph.graph['additional_outputs'] = ['Loop', 'Unsqueeze']

        AddOutputRecursive().find_and_replace_pattern(main_graph)
        loop_node = Node(main_graph, 'Loop')
        self.assertEqual(len(loop_node.output_port_map), 2)
        self.assertEqual(len(loop_node.out_ports()), 2)
        self.assertEqual(loop_node.out_port(2).get_destination().node.op, 'Result')
        unsq_node = Node(sub_graph_1, 'Unsqueeze')
        self.assertEqual(len(unsq_node.out_ports()), 1)
        self.assertEqual(unsq_node.out_port(0).get_destination().node.op, 'Result')
