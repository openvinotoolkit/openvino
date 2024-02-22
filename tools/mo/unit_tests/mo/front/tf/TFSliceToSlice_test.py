# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.front.tf.TFSliceToSlice import TFSliceToSliceReplacer
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, regular_op_with_empty_data, result, const, connect_front

nodes = {
    **regular_op_with_empty_data('input', {'type': 'Parameter'}),
    **regular_op_with_empty_data('tfslice', {'op': 'TFSlice', 'type': None}),
    **const('begin', np.array(0)),
    **const('size', np.array([-1])),
    **regular_op_with_empty_data('john_doe', {'op': 'Sum', 'type': None}),
    **result(),

    # nodes after replacement
    **const('minus_one', np.array(-1)),
    **regular_op_with_empty_data('shapeof', {'op': 'ShapeOf', 'type': 'ShapeOf'}),
    **regular_op_with_empty_data('end_const', {'op': 'Add', 'type': 'Add'}),
    **regular_op_with_empty_data('equal', {'op': 'Equal', 'type': 'Equal'}),
    **regular_op_with_empty_data('select', {'op': 'Select', 'type': 'Select'}),
    **regular_op_with_empty_data('slice', {'op': 'Slice', 'type': None}),
    **regular_op_with_empty_data('cast', {'op': 'Cast', 'type': 'Convert'}),
}


class SliceReplacerTest(unittest.TestCase):
    def test_slice_replacer(self):
        graph = build_graph(nodes_attrs=nodes, edges=[
            *connect_front('input:0', '0:tfslice'),
            *connect_front('begin:0', '1:tfslice'),
            *connect_front('size:0', '2:tfslice'),
            *connect_front('tfslice:0', 'output'),
        ], nodes_with_edges_only=True)
        graph.stage = 'front'

        TFSliceToSliceReplacer().find_and_replace_pattern(graph)

        graph_ref = build_graph(nodes_attrs=nodes, edges=[
            *connect_front('input:0', 'slice'),
            *connect_front('input:0', 'shapeof'),
            *connect_front('begin:0', '1:slice'),
            *connect_front('begin:0', '0:end_const'),
            *connect_front('size:0', '1:end_const'),
            *connect_front('size:0', '0:equal'),
            *connect_front('shapeof:0', '1:select'),
            *connect_front('minus_one:0', '1:equal'),
            *connect_front('equal:0', '0:select'),
            *connect_front('end_const:0', '0:cast'),
            *connect_front('cast:0', '2:select'),
            *connect_front('select:0', '2:slice'),
            *connect_front('slice:0', 'output'),
        ], nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)
