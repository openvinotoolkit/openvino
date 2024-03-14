# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import unittest

from openvino.tools.mo.back.add_outputs_recursive import AddOutputRecursive
from openvino.tools.mo.ops.If import If
from openvino.tools.mo.ops.loop import Loop
from openvino.tools.mo.ops.tensor_iterator import TensorIterator
from openvino.tools.mo.front.common.partial_infer.elemental import copy_shape_infer
from openvino.tools.mo.front.common.partial_infer.utils import int64_array, dynamic_dimension_value, shape_array
from openvino.tools.mo.graph.graph import Node
from unit_tests.utils.graph import build_graph, regular_op_with_empty_data, result, connect, shaped_parameter, \
    valued_const_with_data, shaped_const_with_data, regular_op_with_shaped_data

# test for Loop
main_graph_nodes = {
    **shaped_parameter("IN_1", [1, 4, 64, 54]),
    **shaped_parameter("IN_2", [1, 4, 64, 54]),
    **valued_const_with_data("M", int64_array([5])),
    **valued_const_with_data("cond", int64_array([1])),
    **regular_op_with_empty_data("Loop", {'op': "Loop", 'type': 'Loop', 'sub_graphs': ['body'], "body": None,
                                          'input_port_map': [{'external_port_id': 1, 'internal_layer_id': 2,
                                                              'axis': None},
                                                             {'external_port_id': 2, 'internal_layer_id': 0,
                                                              'axis': None},
                                                             {'external_port_id': 3, 'internal_layer_id': 1,
                                                              'axis': None}],
                                          'output_port_map': [{'external_port_id': 0, 'internal_layer_id': 4,
                                                               'axis': None},
                                                              {'external_port_id': -1, 'internal_layer_id': 5,
                                                               'axis': None, 'purpose': "execution_condition"}],
                                          'back_edges': [{'from_layer': 8, 'to_layer': 7},
                                                         {'from_layer': 10, 'to_layer': 9}],
                                 'infer': Loop.infer}),
    **result("OUT_1")
}

sub_graph_1_nodes = {
    **shaped_parameter("IN_2", int64_array([1, 4, 64, 54]), {'internal_layer_id': 0}),
    **valued_const_with_data("M_2", int64_array([10])),
    **valued_const_with_data("cond_2", int64_array([1])),
    **regular_op_with_empty_data("Loop_2", {'op': "Loop", 'type': 'Loop', 'sub_graphs': ['body'], "body": None,
                                            'input_port_map': [{'external_port_id': 1, 'internal_layer_id': 0,
                                                                'axis': None},
                                                               {'external_port_id': 2, 'internal_layer_id': 2,
                                                                'axis': None}],
                                            'output_port_map': [{'external_port_id': 0, 'internal_layer_id': 7,
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
    **regular_op_with_empty_data("in_1_int_out",
                                 {'op': 'Result', 'type': 'Result', 'infer': lambda x: None, 'internal_layer_id': 4}),
    **shaped_parameter("cond_1_int", int64_array([1]), {'internal_layer_id': 2}),
    **regular_op_with_empty_data("cond_1_int_out", {'op': 'Result', 'type': 'Result', 'infer': lambda x: None,
                                                    'internal_layer_id': 5}),
}

sub_graph_2_nodes = {
    **shaped_parameter('cond_2_int', [1, 4, 64, 54], {'internal_layer_id': 0}),
    **regular_op_with_empty_data("cond_2_int_out",
                                 {'op': 'Result', 'type': 'Result', 'infer': lambda x: None, 'internal_layer_id': 8}),
    **shaped_parameter('in_2_int', [1, 4, 64, 54], {'internal_layer_id': 1}),
    **shaped_const_with_data('ones', int64_array([1, 4, 64, 54]), {'internal_layer_id': 9}),
    **regular_op_with_shaped_data('OUT_2', int64_array([1, 4, 64, 54]), {'op': "Add", 'infer': copy_shape_infer}),
    **regular_op_with_empty_data('OUT_2_out',
                                 {'op': 'Result', 'type': 'Result', 'infer': lambda x: None, 'internal_layer_id': 7}),
    **regular_op_with_shaped_data('in_2_int_out', int64_array([1, 4, 64, 54]),
                                  {'op': 'Result', 'type': 'Result', 'infer': lambda x: None, 'internal_layer_id': 6})
}


def ti_create_main_graph(body):
    main_graph = build_graph(nodes_attrs=ti_main_graph_nodes,
                             edges=[*connect('M', '0:Loop'),
                                    *connect('cond', '1:Loop'),
                                    *connect('IN_2', '2:Loop'),
                                    *connect('IN_1', "3:Loop"),
                                    *connect('Loop:0', 'OUT_1')],
                             nodes_with_edges_only=True)
    loop_node = Node(main_graph, 'Loop')
    loop_node.body = body
    loop_node.in_edge(0)['external_port_id'] = 0
    loop_node.in_edge(1)['external_port_id'] = 1
    loop_node.in_edge(2)['external_port_id'] = 2
    loop_node.in_edge(3)['external_port_id'] = 3
    loop_node.out_edge(0)['external_port_id'] = 4

    return main_graph


def if_create_main_graph():
    sub_graph_2 = build_graph(nodes_attrs=if_sub_graph_2_then_nodes,
                              edges=[*connect('in_2_int', 'OUT_2'),
                                     *connect('ones', 'OUT_2'),
                                     *connect('OUT_2', 'OUT_2_out')],
                              nodes_with_edges_only=True)

    sub_graph_2_else = build_graph(nodes_attrs=if_sub_graph_2_else_nodes,
                                   edges=[*connect('in_2_int_else', 'OUT_2_else'),
                                          *connect('ones_else', 'OUT_2_else'),
                                          *connect('OUT_2_else', 'OUT_2_out_else')],
                                   nodes_with_edges_only=True)

    sub_graph_1 = build_graph(nodes_attrs=if_sub_graph_1_then_nodes,
                              edges=[*connect('cond_2', '0:If_2'),
                                     *connect('IN_2', '1:If_2'),
                                     *connect('If_2:0', 'If_2_out'),
                                     *connect('in_1_int', 'in_1_int_out')],
                              nodes_with_edges_only=True)
    if_node_1 = Node(sub_graph_1, 'If_2')
    if_node_1.then_graph = sub_graph_2
    if_node_1.else_graph = sub_graph_2_else

    return sub_graph_1


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
                                         *connect('Loop_2:0', 'Loop_2_out'),
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
                                        *connect('Loop:0', 'OUT_1')],
                                 nodes_with_edges_only=True)
        loop_node = Node(main_graph, 'Loop')
        loop_node.body = sub_graph_1
        main_graph.graph['additional_outputs'] = ['Loop', 'Loop_2']
        loop_node_1['out_ports_count'] = 2
        loop_node_1.add_output_port(1)
        loop_node_1['output_port_map'].append({'external_port_id': 1, 'internal_layer_id': 8, 'axis': None})

        loop_node_output_port_map_len = len(loop_node.output_port_map)
        loop_node_out_ports_len = len(loop_node.out_ports())
        loop_2_out_ports_len = len(loop_node_1.out_ports())
        max_layer_id = 5

        results = AddOutputRecursive().find_and_replace_pattern(main_graph)

        self.assertEqual(len(results), 2)
        loop_node = Node(main_graph, 'Loop')
        self.assertEqual(len(loop_node.output_port_map), loop_node_output_port_map_len + 2)
        self.assertEqual(len(loop_node.out_ports()), loop_node_out_ports_len + 2)
        self.assertEqual(loop_node.out_port(1).get_destination().node.op, 'Result')
        self.assertTrue(np.all(loop_node.out_port(1).data.get_shape() == int64_array([5, 10, 4, 64, 54])))
        last_node = Node(sub_graph_1, 'Loop_2')
        self.assertEqual(len(last_node.out_ports()), loop_2_out_ports_len)
        unsq_node = last_node.out_port(0).get_destinations()[1].node
        self.assertEqual(unsq_node.op, 'Unsqueeze')
        self.assertEqual(unsq_node.out_port(0).get_destination().node.op, 'Result')
        self.assertEqual(unsq_node.out_port(0).get_destination().node.internal_layer_id, max_layer_id + 3)
        self.assertTrue(np.all(unsq_node.out_port(0).data.get_shape() == int64_array([1, 10, 4, 64, 54])))


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
                                          'output_port_map': [{'external_port_id': 4, 'internal_layer_id': 4, 'axis': None}],
                                          'back_edges': [{'from_layer': 8, 'to_layer': 7},
                                                         {'from_layer': 10, 'to_layer': 9}],
                                 'infer': TensorIterator.infer}),
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
    @staticmethod
    def create_graph():
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
                                         *connect('Loop_2:0', 'Loop_2_out'),
                                         *connect('in_1_int', 'in_1_int_out'),
                                         *connect('cond_1_int', 'cond_1_int_out')],
                                  nodes_with_edges_only=True)
        loop_node_1 = Node(sub_graph_1, 'Loop_2')
        loop_node_1.body = sub_graph_2
        loop_node_1.in_edge(0)['external_port_id'] = 0
        loop_node_1.in_edge(1)['external_port_id'] = 1
        loop_node_1.out_edge(0)['external_port_id'] = 2

        main_graph = ti_create_main_graph(sub_graph_1)
        main_graph.graph['additional_outputs'] = ['Loop', 'Loop_2']

        return main_graph, sub_graph_1

    def check_body_last_node(self, body, node_id, loop_2_node_out_ports_len):
        last_node = Node(body, node_id)
        max_layer_id = 5
        self.assertEqual(len(last_node.out_ports()), loop_2_node_out_ports_len)
        unsq_node = last_node.out_port(0).get_destinations()[1].node
        self.assertEqual(unsq_node.op, 'Unsqueeze')
        self.assertEqual(unsq_node.out_port(0).get_destination().node.op, 'Result')
        self.assertEqual(unsq_node.out_port(0).get_destination().node.internal_layer_id, max_layer_id + 3)
        self.assertTrue(np.all(unsq_node.out_port(0).data.get_shape() == int64_array([1, 1, 4, 64, 54])))

    def check_loop_node(self, graph, node_id, port_map_len, out_ports_len):
        loop_node = Node(graph, node_id)
        self.assertEqual(len(loop_node.output_port_map), port_map_len + 1)
        self.assertEqual(len(loop_node.out_ports()), out_ports_len + 1)
        self.assertEqual(loop_node.out_port(1).get_destination().node.op, 'Result')

    def test_add_output_1(self):
        main_graph, sub_graph_1 = self.create_graph()

        loop_node = Node(main_graph, 'Loop')
        loop_node_output_port_map_len = len(loop_node.output_port_map)
        loop_node_out_ports_len = len(loop_node.out_ports())
        loop_node_2 = Node(sub_graph_1, 'Loop_2')
        loop_2_node_out_ports_len = len(loop_node_2.out_ports())

        AddOutputRecursive().find_and_replace_pattern(main_graph)

        self.check_loop_node(main_graph, 'Loop', loop_node_output_port_map_len, loop_node_out_ports_len)
        self.assertTrue(np.all(loop_node.out_port(1).data.get_shape() == int64_array([1, 1, 4, 64, 54])))
        self.check_body_last_node(sub_graph_1, 'Loop_2', loop_2_node_out_ports_len)

    def test_add_output_dynamic(self):
        main_graph, sub_graph_1 = self.create_graph()

        loop_node = Node(main_graph, 'Loop')
        loop_node_output_port_map_len = len(loop_node.output_port_map)
        loop_node_out_ports_len = len(loop_node.out_ports())
        loop_node_2 = Node(sub_graph_1, 'Loop_2')
        loop_2_node_out_ports_len = len(loop_node_2.out_ports())

        loop_node.input_port_map[2]['axis'] = 1
        loop_node.input_port_map[2]['start'] = 0
        loop_node.input_port_map[2]['end'] = -1
        loop_node.input_port_map[2]['stride'] = 1
        in_1_node = Node(main_graph, 'IN_1')
        in_1_node['shape'] = shape_array([1, dynamic_dimension_value, 64, 54])

        AddOutputRecursive().find_and_replace_pattern(main_graph)

        self.check_loop_node(main_graph, 'Loop', loop_node_output_port_map_len, loop_node_out_ports_len)
        self.assertTrue(np.all(loop_node.out_port(1).data.get_shape() ==
                               shape_array([dynamic_dimension_value, 1, 4, 64, 54])))
        self.check_body_last_node(sub_graph_1, 'Loop_2', loop_2_node_out_ports_len)

    def test_add_output_several_iterations(self):
        main_graph, sub_graph_1 = self.create_graph()

        loop_node = Node(main_graph, 'Loop')
        loop_node_output_port_map_len = len(loop_node.output_port_map)
        loop_node_out_ports_len = len(loop_node.out_ports())
        loop_node_2 = Node(sub_graph_1, 'Loop_2')
        loop_2_node_out_ports_len = len(loop_node_2.out_ports())

        loop_node.input_port_map[2]['axis'] = 1
        loop_node.input_port_map[2]['start'] = 0
        loop_node.input_port_map[2]['end'] = -1
        loop_node.input_port_map[2]['stride'] = 1
        loop_node.output_port_map[0]['axis'] = 1
        loop_node.output_port_map[0]['start'] = 0
        loop_node.output_port_map[0]['end'] = 10
        loop_node.output_port_map[0]['stride'] = 2

        AddOutputRecursive().find_and_replace_pattern(main_graph)

        self.check_loop_node(main_graph, 'Loop', loop_node_output_port_map_len, loop_node_out_ports_len)
        self.assertTrue(np.all(loop_node.out_port(1).data.get_shape() == shape_array([4, 1, 4, 64, 54])))
        self.assertTrue(np.all(loop_node.out_port(0).data.get_shape() == shape_array([1, 5, 64, 54])))
        self.check_body_last_node(sub_graph_1, 'Loop_2', loop_2_node_out_ports_len)

    def test_add_output_several_iterations_wo_start_end(self):
        main_graph, sub_graph_1 = self.create_graph()

        loop_node = Node(main_graph, 'Loop')
        loop_node_output_port_map_len = len(loop_node.output_port_map)
        loop_node_out_ports_len = len(loop_node.out_ports())
        loop_node.input_port_map[2]['axis'] = 1
        loop_node.input_port_map[2]['stride'] = 1

        loop_node_2 = Node(sub_graph_1, 'Loop_2')
        loop_2_node_out_ports_len = len(loop_node_2.out_ports())

        AddOutputRecursive().find_and_replace_pattern(main_graph)

        self.check_loop_node(main_graph, 'Loop', loop_node_output_port_map_len, loop_node_out_ports_len)
        self.assertTrue(np.all(loop_node.out_port(1).data.get_shape() == shape_array([4, 1, 4, 64, 54])))
        self.check_body_last_node(sub_graph_1, 'Loop_2', loop_2_node_out_ports_len)

    def test_add_output_several_iterations_negative_end(self):
        main_graph, sub_graph_1 = self.create_graph()

        loop_node = Node(main_graph, 'Loop')
        loop_node_output_port_map_len = len(loop_node.output_port_map)
        loop_node_out_ports_len = len(loop_node.out_ports())
        loop_node_2 = Node(sub_graph_1, 'Loop_2')
        loop_2_node_out_ports_len = len(loop_node_2.out_ports())

        loop_node.input_port_map[2]['axis'] = 1
        loop_node.input_port_map[2]['start'] = 0
        loop_node.input_port_map[2]['end'] = -3
        loop_node.input_port_map[2]['stride'] = 1
        loop_node.output_port_map[0]['axis'] = 1
        loop_node.output_port_map[0]['start'] = 0
        loop_node.output_port_map[0]['end'] = -1
        loop_node.output_port_map[0]['stride'] = 2

        AddOutputRecursive().find_and_replace_pattern(main_graph)

        self.check_loop_node(main_graph, 'Loop', loop_node_output_port_map_len, loop_node_out_ports_len)
        self.assertTrue(np.all(loop_node.out_port(1).data.get_shape() == shape_array([2, 1, 4, 64, 54])))
        self.assertTrue(np.all(loop_node.out_port(0).data.get_shape() == shape_array([1, 2, 64, 54])))
        self.check_body_last_node(sub_graph_1, 'Loop_2', loop_2_node_out_ports_len)

    def test_add_output_several_iterations_negative_stride(self):
        main_graph, sub_graph_1 = self.create_graph()

        loop_node = Node(main_graph, 'Loop')

        loop_node_output_port_map_len = len(loop_node.output_port_map)
        loop_node_out_ports_len = len(loop_node.out_ports())
        loop_node_2 = Node(sub_graph_1, 'Loop_2')
        loop_2_node_out_ports_len = len(loop_node_2.out_ports())

        loop_node.input_port_map[2]['axis'] = 1
        loop_node.input_port_map[2]['start'] = -1
        loop_node.input_port_map[2]['end'] = 0
        loop_node.input_port_map[2]['stride'] = -2
        loop_node.output_port_map[0]['axis'] = 1
        loop_node.output_port_map[0]['start'] = 0
        loop_node.output_port_map[0]['end'] = -1
        loop_node.output_port_map[0]['stride'] = 2

        AddOutputRecursive().find_and_replace_pattern(main_graph)

        self.check_loop_node(main_graph, 'Loop', loop_node_output_port_map_len, loop_node_out_ports_len)
        self.assertTrue(np.all(loop_node.out_port(1).data.get_shape() == shape_array([2, 1, 4, 64, 54])))
        self.assertTrue(np.all(loop_node.out_port(0).data.get_shape() == shape_array([1, 2, 64, 54])))
        self.check_body_last_node(sub_graph_1, 'Loop_2', loop_2_node_out_ports_len)

    def test_add_output_several_iterations_negative_start_end_input(self):
        main_graph, sub_graph_1 = self.create_graph()

        loop_node = Node(main_graph, 'Loop')
        loop_node_output_port_map_len = len(loop_node.output_port_map)
        loop_node_out_ports_len = len(loop_node.out_ports())
        loop_node_2 = Node(sub_graph_1, 'Loop_2')
        loop_2_node_out_ports_len = len(loop_node_2.out_ports())

        loop_node.input_port_map[2]['axis'] = 1
        loop_node.input_port_map[2]['start'] = -1
        loop_node.input_port_map[2]['end'] = -4
        loop_node.input_port_map[2]['stride'] = -2
        loop_node.output_port_map[0]['axis'] = 1
        loop_node.output_port_map[0]['start'] = 0
        loop_node.output_port_map[0]['end'] = -1
        loop_node.output_port_map[0]['stride'] = 2

        AddOutputRecursive().find_and_replace_pattern(main_graph)

        self.check_loop_node(main_graph, 'Loop', loop_node_output_port_map_len, loop_node_out_ports_len)
        self.assertTrue(np.all(loop_node.out_port(1).data.get_shape() == shape_array([2, 1, 4, 64, 54])))
        self.assertTrue(np.all(loop_node.out_port(0).data.get_shape() == shape_array([1, 2, 64, 54])))
        self.check_body_last_node(sub_graph_1, 'Loop_2', loop_2_node_out_ports_len)

    def test_add_output_several_iterations_negative_start_end_output(self):
        main_graph, sub_graph_1 = self.create_graph()

        loop_node = Node(main_graph, 'Loop')
        loop_node_output_port_map_len = len(loop_node.output_port_map)
        loop_node_out_ports_len = len(loop_node.out_ports())
        loop_node_2 = Node(sub_graph_1, 'Loop_2')
        loop_2_node_out_ports_len = len(loop_node_2.out_ports())

        loop_node.input_port_map[2]['axis'] = 1
        loop_node.input_port_map[2]['start'] = -1
        loop_node.input_port_map[2]['end'] = -4
        loop_node.input_port_map[2]['stride'] = -2
        loop_node.output_port_map[0]['axis'] = 1
        loop_node.output_port_map[0]['start'] = -4
        loop_node.output_port_map[0]['end'] = -1
        loop_node.output_port_map[0]['stride'] = 1

        AddOutputRecursive().find_and_replace_pattern(main_graph)

        self.check_loop_node(main_graph, 'Loop', loop_node_output_port_map_len, loop_node_out_ports_len)
        self.assertTrue(np.all(loop_node.out_port(1).data.get_shape() == shape_array([2, 1, 4, 64, 54])))
        self.assertTrue(np.all(loop_node.out_port(0).data.get_shape() == shape_array([1, 3, 64, 54])))
        self.check_body_last_node(sub_graph_1, 'Loop_2', loop_2_node_out_ports_len)


# test for If
if_main_graph_nodes = {
    **shaped_parameter("IN_1", [1, 4, 64, 54]),
    **shaped_parameter("IN_2", [1, 4, 64, 54]),
    **valued_const_with_data("cond", int64_array([1])),
    **regular_op_with_empty_data("If", {'op': "If", 'type': 'If', 'sub_graphs': ['then_graph', 'else_graph'],
                                        "then_graph": None, 'else_graph': None, 'infer': If.infer}),
    **result("OUT_1")
}

if_sub_graph_1_then_nodes = {
    **shaped_parameter("IN_2", int64_array([1, 4, 64, 54]), {'input_id': 2}),
    **valued_const_with_data("cond_2", int64_array([1])),
    **regular_op_with_empty_data("If_2", {'op': "If", 'type': 'If', 'sub_graphs': ['then_graph', 'else_graph'],
                                          "then_graph": None, 'else_graph': None, 'infer': If.infer}),
    **regular_op_with_empty_data('If_2_out', {'op': 'Result', 'type': 'Result', 'infer': lambda x: None}),
    **shaped_parameter("in_1_int", int64_array([1, 4, 64, 54]), {'input_id': 1}),
    **regular_op_with_empty_data("in_1_int_out", {'op': 'Result', 'type': 'Result', 'output_id': 0})
}

if_sub_graph_1_else_nodes = {
    **shaped_parameter("in_1_int", int64_array([1, 4, 64, 54]), {'input_id': 1}),
    **regular_op_with_empty_data("in_1_int_out", {'op': 'Result', 'type': 'Result', 'output_id': 0})
}

if_sub_graph_2_then_nodes = {
    **shaped_parameter('in_2_int', [1, 4, 64, 54], {'input_id': 1}),
    **shaped_const_with_data('ones', int64_array([1, 4, 64, 54])),
    **regular_op_with_shaped_data('OUT_2', int64_array([1, 4, 64, 54]), {'op': "Add"}),
    **regular_op_with_empty_data('OUT_2_out', {'op': 'Result', 'type': 'Result', 'output_id': 0}),
}

if_sub_graph_2_else_nodes = {
    **shaped_parameter('in_2_int_else', [1, 4, 64, 54], {'input_id': 1}),
    **shaped_const_with_data('ones_else', int64_array([1, 4, 64, 54])),
    **regular_op_with_shaped_data('OUT_2_else', int64_array([1, 4, 64, 54]), {'op': "Sub"}),
    **regular_op_with_empty_data('OUT_2_out_else', {'op': 'Result', 'type': 'Result', 'output_id': 0}),
}


class IF_AddOutputRecursiveTest(unittest.TestCase):
    def test_add_output_1(self):
        sub_graph_1 = if_create_main_graph()
        if_node_1 = Node(sub_graph_1, 'If_2')

        sub_graph_1_else = build_graph(nodes_attrs=if_sub_graph_1_else_nodes,
                                       edges=[*connect('in_1_int', 'in_1_int_out')],
                                       nodes_with_edges_only=True)

        main_graph = build_graph(nodes_attrs=if_main_graph_nodes,
                                 edges=[*connect('cond', '0:If'),
                                        *connect('IN_1', '1:If'),
                                        *connect('IN_2', "2:If"),
                                        *connect('If:0', 'OUT_1')],
                                 nodes_with_edges_only=True)
        if_node = Node(main_graph, 'If')
        if_node.then_graph = sub_graph_1
        if_node.else_graph = sub_graph_1_else
        if_node_out_ports_len = len(if_node.out_ports())
        if_2_node_out_ports_len = len(if_node_1.out_ports())

        main_graph.graph['additional_outputs'] = ['If', ['If_2', 'in_1_int']]

        AddOutputRecursive().find_and_replace_pattern(main_graph)
        if_node = Node(main_graph, 'If')
        self.assertEqual(len(if_node.out_ports()), if_node_out_ports_len + 1)
        self.assertEqual(if_node.out_port(1).get_destination().node.op, 'Result')
        self.assertTrue(np.all(if_node.out_port(1).data.get_shape() == int64_array([1, 4, 64, 54])))
        last_node = Node(sub_graph_1, 'If_2')
        self.assertEqual(len(last_node.out_ports()), if_2_node_out_ports_len)
        self.assertEqual(last_node.out_port(0).get_destinations()[1].node.op, 'Result')
        self.assertTrue(np.all(last_node.out_port(0).data.get_shape() == int64_array([1, 4, 64, 54])))


class SplitUserPathTest(unittest.TestCase):

    @staticmethod
    def create_graph():
        sub_graph_1 = if_create_main_graph()
        out_node = Node(sub_graph_1, 'If_2_out')
        out_node['internal_layer_id'] = 4

        main_graph = ti_create_main_graph(sub_graph_1)

        return main_graph

    def test_linear_graph_change(self):
        graph = self.create_graph()
        path = ['Loop', 'in_1_int']
        ref_path = []
        loop_node = Node(graph, 'Loop')
        ref_path.append({'node': loop_node, 'graph': graph})
        ref_path.append({'node': Node(loop_node.body, 'in_1_int'), 'graph': loop_node.body})

        tracks = AddOutputRecursive().split_path_to_simple_tracks(graph, path)

        self.assertTrue(np.all(tracks[0] == ref_path))

    def test_1_if_graph_change(self):
        graph = self.create_graph()
        path = ['Loop', 'If_2', ['OUT_2', 'OUT_2_else']]
        ref_path = [[]]
        loop_node = Node(graph, 'Loop')
        ref_path[0].append({'node': loop_node, 'graph': graph})
        if_node = Node(loop_node.body, 'If_2')
        ref_path[0].append({'node': if_node, 'graph': loop_node.body})
        ref_path.append([])
        ref_path[1] = ref_path[0][:]
        ref_path[0].append({'node': Node(if_node.then_graph, 'OUT_2'), 'graph': if_node.then_graph})
        ref_path[1].append({'node': Node(if_node.else_graph, 'OUT_2_else'), 'graph': if_node.else_graph})

        tracks = AddOutputRecursive().split_path_to_simple_tracks(graph, path)

        self.assertTrue(np.all(tracks[0] == ref_path[0]))
        self.assertTrue(np.all(tracks[1] == ref_path[1]))

    def test_1_if_graph_change_add_output(self):
        graph = self.create_graph()
        graph.graph['additional_outputs'] = ['Loop', 'If_2', ['OUT_2', 'OUT_2_else']]

        AddOutputRecursive().find_and_replace_pattern(graph)

        loop_node = Node(graph, 'Loop')
        if_node = Node(loop_node.body, 'If_2')
        left_node = Node(if_node.then_graph, 'OUT_2')
        right_node = Node(if_node.else_graph, 'OUT_2_else')
        self.assertEqual(len(left_node.out_port(0).get_destinations()), 2)
        self.assertEqual(left_node.out_port(0).get_destinations()[1].node.op, 'Result')

        self.assertEqual(len(right_node.out_port(0).get_destinations()), 2)
        self.assertEqual(right_node.out_port(0).get_destinations()[1].node.op, 'Result')

        self.assertTrue(len(if_node.out_ports()), 2)
        self.assertTrue(if_node.out_port(1).get_destination().node.op, 'Result')

        self.assertTrue(len(loop_node.out_ports()), 2)
        self.assertTrue(loop_node.out_port(1).get_destination().node.op, 'Result')
