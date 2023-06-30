# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from argparse import Namespace

import numpy as np
from generator import generator, generate

from openvino.tools.mo.back.ChangeRandomUniformOutputType import ChangeRandomUniformOutputType
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, result, connect, regular_op_with_shaped_data

nodes = {
    **regular_op_with_shaped_data('placeholder', [3], {'type': 'Parameter'}),
    **regular_op_with_shaped_data('random_uniform', [3, 4, 5], {'type': 'RandomUniform', 'op': 'RandomUniform'}),
    **regular_op_with_shaped_data('convert', [3, 4, 5], {'type': 'Convert'}),
    **result('result'),

    # new RandomUniform node and inputs
    **regular_op_with_shaped_data('min_val', [1], {'type': 'Const'}),
    **regular_op_with_shaped_data('max_val', [1], {'type': 'Const'}),
    **regular_op_with_shaped_data('shape', [3], {'type': 'Const'}),
}

edges = [*connect('placeholder', '0:random_uniform'), *connect('min_val', '1:random_uniform'),
         *connect('max_val', '2:random_uniform'), *connect('random_uniform', 'result')]
edges_with_convert = [*connect('placeholder', '0:random_uniform'), *connect('min_val', '1:random_uniform'),
                      *connect('max_val', '2:random_uniform'), *connect('random_uniform', 'convert'),
                      *connect('convert', 'result'), ]


class ChangeRandomUniformOutputTypeTest(unittest.TestCase):
    def test_change_random_uniform_output_type_case1(self):
        ir_type, out_type, dst_type = "FP16", np.float32, np.float16
        graph = build_graph(nodes, edges, cli=Namespace(data_type=ir_type))
        graph_ref = build_graph(nodes, edges if dst_type is None else edges_with_convert, {},
                                nodes_with_edges_only=True)
        Node(graph, 'random_uniform')['output_type'] = out_type

        ChangeRandomUniformOutputType().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)

        if dst_type is not None:
            convert_node = Node(graph, 'random_uniform').out_port(0).get_destination().node
            self.assertTrue(convert_node['dst_type'] == dst_type)

    def test_change_random_uniform_output_type_case2(self):
        ir_type, out_type, dst_type = "FP32", np.float16, np.float32
        graph = build_graph(nodes, edges, cli=Namespace(data_type=ir_type))
        graph_ref = build_graph(nodes, edges if dst_type is None else edges_with_convert, {},
                                nodes_with_edges_only=True)
        Node(graph, 'random_uniform')['output_type'] = out_type

        ChangeRandomUniformOutputType().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)

        if dst_type is not None:
            convert_node = Node(graph, 'random_uniform').out_port(0).get_destination().node
            self.assertTrue(convert_node['dst_type'] == dst_type)

    def test_change_random_uniform_output_type_case3(self):
        ir_type, out_type, dst_type = "FP32", np.float32, None
        graph = build_graph(nodes, edges, cli=Namespace(data_type=ir_type))
        graph_ref = build_graph(nodes, edges if dst_type is None else edges_with_convert, {},
                                nodes_with_edges_only=True)
        Node(graph, 'random_uniform')['output_type'] = out_type

        ChangeRandomUniformOutputType().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)

        if dst_type is not None:
            convert_node = Node(graph, 'random_uniform').out_port(0).get_destination().node
            self.assertTrue(convert_node['dst_type'] == dst_type)

    def test_change_random_uniform_output_type_case4(self):
        ir_type, out_type, dst_type = "FP32", np.int64, None
        graph = build_graph(nodes, edges, cli=Namespace(data_type=ir_type))
        graph_ref = build_graph(nodes, edges if dst_type is None else edges_with_convert, {},
                                nodes_with_edges_only=True)
        Node(graph, 'random_uniform')['output_type'] = out_type

        ChangeRandomUniformOutputType().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)

        if dst_type is not None:
            convert_node = Node(graph, 'random_uniform').out_port(0).get_destination().node
            self.assertTrue(convert_node['dst_type'] == dst_type)
   
