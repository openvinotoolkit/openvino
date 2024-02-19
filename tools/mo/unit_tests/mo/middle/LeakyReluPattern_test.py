# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.middle.LeakyReluPattern import LeakyReLUFusion
from openvino.tools.mo.front.common.partial_infer.utils import float_array, int64_array
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.ops.result import Result
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, result, build_graph_with_edge_attrs, connect, \
    regular_op_with_shaped_data, valued_const_with_data, connect_data

shape = int64_array([1, 3, 5, 2])
nodes = {**regular_op_with_shaped_data('input', shape, {'type': 'Parameter', 'op': 'Parameter'}),
         **regular_op_with_shaped_data('mul', shape, {'type': 'Multiply', 'name': 'mul'}),
         **regular_op_with_shaped_data('max', shape, {'type': 'Maximum', 'name': 'final_max'}),
         **valued_const_with_data('const', float_array([0.5])),
         **result('result')
         }

edges = [*connect('input:0', '0:mul'),
         *connect('const', '1:mul'),
         *connect_data('input', '0:max'),
         *connect('mul:0', '1:max'),
         *connect('max:0', 'result'),
         ]

ref_nodes = {**regular_op_with_shaped_data('input', shape, {'type': 'Parameter', 'op': 'Parameter'}),
             **regular_op_with_shaped_data('leaky_relu', shape, {'type': 'LeakyReLU', 'name': 'max_final',
                                                                 'negative_slope': None}),
             **result('result')
         }
ref_edges = [*connect('input:0', 'leaky_relu'), *connect('leaky_relu', 'result')]


class LeakyReluFusionTest(unittest.TestCase):
    def test_leaky_relu_data_port_0(self):
        graph = build_graph_with_edge_attrs(nodes, edges, {})
        graph_ref = build_graph(ref_nodes, ref_edges)
        Node(graph_ref, 'leaky_relu')['negative_slope'] = 0.5

        LeakyReLUFusion().find_and_replace_pattern(graph)
        graph.clean_up()

        (flag, resp) = compare_graphs(graph, graph_ref, 'result')
        self.assertTrue(flag, resp)
        self.assertTrue(len(graph.get_op_nodes(name='final_max')) == 1 and
                        graph.get_op_nodes(name='final_max')[0].op == 'LeakyReLU')

    def test_leaky_relu_not_applicable_non_scalar_const(self):
        # const value is not a scalar or 1D tensor with 1 element so the transformation is not applicable
        graph = build_graph_with_edge_attrs(nodes, edges, {})
        Node(graph, 'const')['value'] = float_array([0.5, 0.7])
        Node(graph, 'const_d')['value'] = float_array([0.5, 0.7])
        graph_ref = graph.copy()

        LeakyReLUFusion().find_and_replace_pattern(graph)
        graph.clean_up()

        (flag, resp) = compare_graphs(graph, graph_ref, 'result')
        self.assertTrue(flag, resp)

    def test_leaky_relu_mul_multiple_consumers(self):
        # multiple consumers of Mul operation
        graph = build_graph_with_edge_attrs(nodes, edges, {})
        additional_result = Result(graph, {'name': 'result_2'}).create_node()
        Node(graph, 'mul').out_port(0).connect(additional_result.in_port(0))

        ref_nodes = {**regular_op_with_shaped_data('input', shape, {'type': 'Parameter', 'op': 'Parameter'}),
                     **regular_op_with_shaped_data('mul', shape, {'type': 'Multiply', 'name': 'mul'}),
                     **regular_op_with_shaped_data('max', shape, {'type': 'Maximum', 'name': 'final_max'}),
                     **valued_const_with_data('const', float_array([0.5])),
                     **regular_op_with_shaped_data('leaky_relu', shape, {'type': 'LeakyReLU', 'name': 'max_final',
                                                                         'negative_slope': None}),
                     **result('result'),
                     **result('result_2')
                     }
        ref_edges = [*connect('input:0', '0:mul'),
                     *connect('const', '1:mul'),
                     *connect('max:0', 'result'),
                     *connect('mul:0', 'result_2'),
                     *connect_data('input', 'leaky_relu'),
                     *connect('leaky_relu', 'result')
                     ]
        graph_ref = build_graph_with_edge_attrs(ref_nodes, ref_edges)

        LeakyReLUFusion().find_and_replace_pattern(graph)
        graph.clean_up()

        (flag, resp) = compare_graphs(graph, graph_ref, 'result')
        self.assertTrue(flag, resp)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result_2')
        self.assertTrue(flag, resp)
