# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from argparse import Namespace

import numpy as np

from extensions.back.ChangeRandomUniformOutputType import ChangeRandomUniformOutputType
from mo.graph.graph import Node
from mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, result, connect, regular_op_with_shaped_data

nodes = {
    **regular_op_with_shaped_data('placeholder', [3], {'type': 'Parameter'}),
    **regular_op_with_shaped_data('random_uniform', [3, 4, 5], {'type': 'RandomUniform', 'op': 'RandomUniform'}),
    **regular_op_with_shaped_data('convert', [3, 4, 5], {'type': 'Convert'}),
    **result('result'),

    # new Roll node and inputs
    **regular_op_with_shaped_data('min_val', [1], {'type': 'Const'}),
    **regular_op_with_shaped_data('max_val', [1], {'type': 'Const'}),
    **regular_op_with_shaped_data('shape', [3], {'type': 'Const'}),
}


class ChangeRandomUniformOutputTypeTest(unittest.TestCase):
    def test_fp32_to_fp16(self):
        graph = build_graph(nodes,
                            [*connect('placeholder', '0:random_uniform'),
                             *connect('min_val', '1:random_uniform'),
                             *connect('max_val', '2:random_uniform'),
                             *connect('random_uniform', 'result'),
                             ], cli=Namespace(data_type="FP16"))

        graph_ref = build_graph(nodes,
                                [*connect('placeholder', '0:random_uniform'),
                                 *connect('min_val', '1:random_uniform'),
                                 *connect('max_val', '2:random_uniform'),
                                 *connect('random_uniform', 'convert'),
                                 *connect('convert', 'result'),
                                 ], {}, nodes_with_edges_only=True)

        Node(graph, 'random_uniform')['output_type'] = np.float32

        ChangeRandomUniformOutputType().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)

        convert_node = Node(graph, 'random_uniform').out_port(0).get_destination().node
        self.assertTrue(convert_node['dst_type'] == np.float16)

    def test_fp16_to_fp32(self):
        graph = build_graph(nodes,
                            [*connect('placeholder', '0:random_uniform'),
                             *connect('min_val', '1:random_uniform'),
                             *connect('max_val', '2:random_uniform'),
                             *connect('random_uniform', 'result'),
                             ], cli=Namespace(data_type="FP32"))

        graph_ref = build_graph(nodes,
                                [*connect('placeholder', '0:random_uniform'),
                                 *connect('min_val', '1:random_uniform'),
                                 *connect('max_val', '2:random_uniform'),
                                 *connect('random_uniform', 'convert'),
                                 *connect('convert', 'result'),
                                 ], {}, nodes_with_edges_only=True)

        Node(graph, 'random_uniform')['output_type'] = np.float16

        ChangeRandomUniformOutputType().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)

        convert_node = Node(graph, 'random_uniform').out_port(0).get_destination().node
        self.assertTrue(convert_node['dst_type'] == np.float32)

    def test_no_convert(self):
        graph = build_graph(nodes,
                            [*connect('placeholder', '0:random_uniform'),
                             *connect('min_val', '1:random_uniform'),
                             *connect('max_val', '2:random_uniform'),
                             *connect('random_uniform', 'result'),
                             ], cli=Namespace(data_type="FP32"))

        graph_ref = build_graph(nodes,
                                [*connect('placeholder', '0:random_uniform'),
                                 *connect('min_val', '1:random_uniform'),
                                 *connect('max_val', '2:random_uniform'),
                                 *connect('random_uniform', 'result'),
                                 ], {}, nodes_with_edges_only=True)

        Node(graph, 'random_uniform')['output_type'] = np.float32

        ChangeRandomUniformOutputType().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_no_convert_case2(self):
        graph = build_graph(nodes,
                            [*connect('placeholder', '0:random_uniform'),
                             *connect('min_val', '1:random_uniform'),
                             *connect('max_val', '2:random_uniform'),
                             *connect('random_uniform', 'result'),
                             ], cli=Namespace(data_type="FP32"))

        graph_ref = build_graph(nodes,
                                [*connect('placeholder', '0:random_uniform'),
                                 *connect('min_val', '1:random_uniform'),
                                 *connect('max_val', '2:random_uniform'),
                                 *connect('random_uniform', 'result'),
                                 ], {}, nodes_with_edges_only=True)

        Node(graph, 'random_uniform')['output_type'] = np.int64

        ChangeRandomUniformOutputType().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)
