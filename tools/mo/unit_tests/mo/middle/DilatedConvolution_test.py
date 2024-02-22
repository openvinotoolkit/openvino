# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.middle.DilatedConvolution import DilatedConvolutionConverter
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, result, connect, \
    regular_op_with_shaped_data, valued_const_with_data

shape = int64_array([1, 375, 500, 24])
nodes = {**regular_op_with_shaped_data('input', shape, {'type': 'Parameter', 'op': 'Parameter'}),
         **valued_const_with_data('stb_bs', int64_array([1, 32, 32, 1])),
         **valued_const_with_data('stb_pad_begin', int64_array([0, 32, 32, 0])),
         **valued_const_with_data('stb_pad_end', int64_array([0, 41, 44, 0])),
         **regular_op_with_shaped_data('space_to_batch', int64_array([1024, 14, 18, 24]),
                                       {'op': 'SpaceToBatch', 'name': 'stb'}),
         **regular_op_with_shaped_data('conv', int64_array([1024, 12, 16, 24]),
                                       {'op': 'Conv2D', 'name': 'conv', 'spatial_dims': int64_array([1, 2]),
                                        'dilation': int64_array([1, 1, 1, 1]),
                                        'pad': int64_array([[0, 0], [0, 0], [0, 0], [0, 0]])}),
         **valued_const_with_data('bts_bs', int64_array([1, 32, 32, 1])),
         **valued_const_with_data('bts_crop_begin', int64_array([0, 0, 0, 0])),
         **valued_const_with_data('bts_crop_end', int64_array([0, 9, 12, 0])),
         **regular_op_with_shaped_data('batch_to_space', shape, {'op': 'BatchToSpace', 'name': 'bts'}),
         **result('result')
         }

edges = [*connect('input', '0:space_to_batch'),
         *connect('stb_bs', '1:space_to_batch'),
         *connect('stb_pad_begin', '2:space_to_batch'),
         *connect('stb_pad_end', '3:space_to_batch'),
         *connect('space_to_batch', '0:conv'),
         *connect('conv', '0:batch_to_space'),
         *connect('bts_bs', '1:batch_to_space'),
         *connect('bts_crop_begin', '2:batch_to_space'),
         *connect('bts_crop_end', '3:batch_to_space'),
         *connect('batch_to_space', 'result')
         ]

ref_nodes = {**regular_op_with_shaped_data('input', shape, {'type': 'Parameter', 'op': 'Parameter'}),
             **regular_op_with_shaped_data('conv', shape,
                                           {'op': 'Conv2D', 'name': 'conv', 'spatial_dims': int64_array([1, 2]),
                                            'dilation': int64_array([1, 32, 32, 1]), 'auto_pad': None,
                                            'pad': int64_array([[0, 0], [32, 32], [32, 32], [0, 0]])}),
             **result('result')
             }
ref_edges = [*connect('input', '0:conv'),
             *connect('conv', 'result')
             ]


class DilatedConvolutionTest(unittest.TestCase):
    def test_dilated_conv_1(self):
        graph = build_graph(nodes, edges)

        graph_ref = build_graph(ref_nodes, ref_edges)

        graph.graph['layout'] = 'NHWC'
        graph.stage = 'middle'

        DilatedConvolutionConverter().find_and_replace_pattern(graph)
        graph.clean_up()

        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)
