# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.middle.RemoveUselessPad import RemoveUselessPad
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, regular_op_with_shaped_data, valued_const_with_data, result, connect, \
    connect_data


class RemoveUselessPadTests(unittest.TestCase):
    def test_useless_pad_constant_input(self):
        nodes = {
            **regular_op_with_shaped_data('placeholder', [1, 10, 20, 3], {'type': 'Parameter'}),
            **regular_op_with_shaped_data('pad', [1, 10, 20, 3], {'type': 'Pad', 'op': 'Pad'}),
            **valued_const_with_data('pads_begin', int64_array([0, 0, 0, 0])),
            **valued_const_with_data('pads_end', int64_array([0, 0, 0, 0])),
            **valued_const_with_data('fill_value', np.array(1)),
            **result('result'),
        }
        edges = [*connect('placeholder', '0:pad'),
                 *connect('pads_begin', '1:pad'),
                 *connect('pads_end', '2:pad'),
                 *connect('fill_value', '3:pad'),
                 *connect('pad', 'result'),
                 ]
        graph = build_graph(nodes, edges)
        RemoveUselessPad().find_and_replace_pattern(graph)
        ref_graph = build_graph(nodes, [*connect('placeholder', 'result')])

        (flag, resp) = compare_graphs(graph, ref_graph, 'result')
        self.assertTrue(flag, resp)

    def test_not_useless_pad_constant_input(self):
        nodes = {
            **regular_op_with_shaped_data('placeholder', [1, 10, 20, 3], {'type': 'Parameter'}),
            **regular_op_with_shaped_data('pad', [1, 10, 20, 3], {'type': 'Pad', 'op': 'Pad'}),
            **valued_const_with_data('pads_begin', int64_array([0, 0, 0, 0])),
            **valued_const_with_data('pads_end', int64_array([0, 1, 0, 0])),
            **valued_const_with_data('fill_value', np.array(1)),
            **result('result'),
        }
        edges = [*connect('placeholder', '0:pad'),
                 *connect('pads_begin', '1:pad'),
                 *connect('pads_end', '2:pad'),
                 *connect('fill_value', '3:pad'),
                 *connect('pad', 'result'),
                 ]
        graph = build_graph(nodes, edges)
        RemoveUselessPad().find_and_replace_pattern(graph)
        ref_graph = build_graph(nodes, edges)

        (flag, resp) = compare_graphs(graph, ref_graph, 'result')
        self.assertTrue(flag, resp)

    def test_not_useless_pad_non_constant_input(self):
        nodes = {
            **regular_op_with_shaped_data('placeholder', [10, 20, 3], {'type': 'Parameter'}),
            **regular_op_with_shaped_data('shape_of_1', [3], {'type': 'ShapeOf'}),
            **regular_op_with_shaped_data('sub', [3], {'type': 'Subtract', 'op': 'Sub'}),
            **valued_const_with_data('desired_output_size', int64_array([10, 20, 3])),
            **regular_op_with_shaped_data('pad', [10, 20, 3], {'type': 'Pad', 'op': 'Pad'}),
            **valued_const_with_data('fill_value', np.array(1)),
            **result('result'),
        }
        edges = [*connect('placeholder', '0:pad'),
                 *connect('placeholder', 'shape_of_1'),
                 *connect('shape_of_1', '0:sub'),
                 *connect('desired_output_size', '1:sub'),
                 *connect('sub', '1:pad'),
                 *connect_data('sub', '2:pad'),
                 *connect('fill_value', '3:pad'),
                 *connect('pad', 'result'),
                 ]
        graph = build_graph(nodes, edges)
        RemoveUselessPad().find_and_replace_pattern(graph)
        ref_graph = build_graph(nodes, edges)

        (flag, resp) = compare_graphs(graph, ref_graph, 'result')
        self.assertTrue(flag, resp)
