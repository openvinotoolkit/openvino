# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.ops.elementwise import Round, Elementwise
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.middle.passes.infer import type_infer
from unit_tests.utils.graph import valued_const_with_data, result, regular_op_with_empty_data, connect, \
    shaped_parameter, build_graph


def round_test_graph(nodes_attributes, value, mode: str):
    graph = build_graph(nodes_attributes,
                        [
                            ('node_1', 'elementwise_node'),
                            ('elementwise_node', 'node_3')
                        ],
                        {
                            'node_1': {
                                'value': value
                            },
                            'elementwise_node': {
                                'op': 'Round',
                                'mode': mode,
                            },
                            'node_3': {
                                'value': None
                            }
                        })
    return graph


class TestElementwiseOp(unittest.TestCase):
    nodes_attributes = {
        'node_1': {
            'shape': np.array([13]),
            'value': None
        },
        'elementwise_node': {
            'op': None,
            'kind': 'op',
            'operation': None
        },
        'node_3': {
            'shape': None
        }
    }

    value = np.array([-23.5, -22.5, -2.5, -1.5, -0.5, 0.5, 0.9, 1.5, 2.3, 2.5, 3.5, 22.5, 23.5])

    def test_elementwise_round_even_infer(self):
        graph = round_test_graph(self.nodes_attributes, self.value, 'half_to_even')

        graph.graph['layout'] = 'NCHW'
        elementwise_node = Node(graph, 'elementwise_node')
        Round.infer(elementwise_node)
        exp_shape = np.array([13])
        res_shape = graph.node['node_3']['shape']
        res_value = graph.node['node_3']['value']
        exp_value = np.array([-24., -22., -2., -2., -0., 0., 1., 2., 2., 2., 4., 22., 24., ])
        for i, value in enumerate(exp_shape):
            self.assertEqual(res_shape[i], value)
        for i, value in enumerate(exp_value):
            self.assertAlmostEqual(res_value[i], value)

    def test_elementwise_round_away_infer(self):
        graph = round_test_graph(self.nodes_attributes, self.value, 'half_away_from_zero')

        graph.graph['layout'] = 'NCHW'
        elementwise_node = Node(graph, 'elementwise_node')
        Round.infer(elementwise_node)
        exp_shape = np.array([13])
        res_shape = graph.node['node_3']['shape']
        res_value = graph.node['node_3']['value']
        exp_value = np.array([-24., -23., -3., -2., -1., 1., 1., 2., 2., 3., 4., 23., 24.])
        for i, value in enumerate(exp_shape):
            self.assertEqual(res_shape[i], value)
        for i, value in enumerate(exp_value):
            self.assertAlmostEqual(res_value[i], value)


class TestElementwiseTypeAlignment(unittest.TestCase):

    @staticmethod
    def build_graph_to_test_type_alignment(edges,
                                           input_1_type=np.float32,
                                           input_2_type=np.float32,
                                           const_type=np.float32):
        input_shape = int64_array([1, 3, 255, 255])
        const_value = np.array([1], dtype=const_type)

        nodes = {
            **shaped_parameter('input_1', input_shape, {'data_type': input_1_type}),
            **shaped_parameter('input_2', input_shape, {'data_type': input_2_type}),
            **regular_op_with_empty_data('add', {'op': 'Add', 'type': 'Add', 'type_infer': Elementwise.type_infer}),
            **valued_const_with_data('const', const_value, kwargs={'data_type': const_type}),
            **result('result'),
        }
        graph = build_graph(nodes, edges, nodes_with_edges_only=True)
        graph.stage = 'back'
        return graph

    def test_first_input_const(self):
        edges = [
            *connect('const', '0:add'),
            *connect('input_1', '1:add'),
            *connect('add', 'result')
        ]
        graph = self.build_graph_to_test_type_alignment(edges, const_type=np.float16, input_1_type=np.float32)

        type_infer(graph)
        const_node = Node(graph, 'const')
        self.assertEqual(const_node.out_port(0).get_data_type(), np.float32)

    def test_second_input_const(self):
        edges = [
            *connect('input_1', '0:add'),
            *connect('const', '1:add'),
            *connect('add', 'result')
        ]
        graph = self.build_graph_to_test_type_alignment(edges, input_1_type=np.float32, const_type=np.float16)

        type_infer(graph)
        const_node = Node(graph, 'const')
        self.assertEqual(const_node.out_port(0).get_data_type(), np.float32)

    def test_raises(self):
        edges = [
            *connect('input_1', '0:add'),
            *connect('input_2', '1:add'),
            *connect('add', 'result')
        ]
        graph = self.build_graph_to_test_type_alignment(edges, input_1_type=np.float32, input_2_type=np.float16)

        self.assertRaises(Exception, type_infer, graph)

    def test_not_raises(self):
        edges = [
            *connect('input_1', '0:add'),
            *connect('input_2', '1:add'),
            *connect('add', 'result')
        ]
        graph = self.build_graph_to_test_type_alignment(edges, input_1_type=np.float32, input_2_type=np.float32)

        type_infer(graph)
        add_node = Node(graph, 'add')
        self.assertEqual(add_node.out_port(0).get_data_type(), np.float32)
