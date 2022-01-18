# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
from generator import generator, generate

from openvino.tools.mo.front.onnx.AttributedSliceToSlice import AttributedSliceToSliceReplacer
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, regular_op_with_empty_data, result, const, connect_front


@generator
class SliceReplacerTest(unittest.TestCase):
    @generate(*[
       {'op': 'AttributedSlice', 'type': None, 'starts': np.array([0, 0]), 'ends': np.array([1, -1]), 'axes': np.array([0, 1])}
    ])
    def test_attributed_slice_replacer(self, attributed_slice_attrs):
        nodes = {
            **regular_op_with_empty_data('input', {'type': 'Parameter'}),
            **regular_op_with_empty_data('attributed_slice', attributed_slice_attrs),
            **result(),

            # nodes after replacement
            **const('start', np.array([0, 0])),
            **const('end', np.array([1, -1])),
            **const('axis', np.array(np.array([0, 1]))),
            **regular_op_with_empty_data('slice', {'op': 'Slice', 'type': None}),
        }

        graph = build_graph(nodes_attrs=nodes, edges=[
            ('input', 'attributed_slice'),
            ('attributed_slice', 'output'),
        ], nodes_with_edges_only=True)
        graph.stage = 'front'

        AttributedSliceToSliceReplacer().find_and_replace_pattern(graph)

        graph_ref = build_graph(nodes_attrs=nodes, edges=[
            ('input', 'slice'),
            *connect_front('start', '1:slice'),
            *connect_front('end', '2:slice'),
            *connect_front('axis', '3:slice'),
            ('slice', 'output'),
        ], nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)
