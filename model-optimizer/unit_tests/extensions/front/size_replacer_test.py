# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
from generator import generator, generate

from extensions.front.SizeReplacer import SizeFrontReplacer
from mo.front.common.partial_infer.utils import int64_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, regular_op_with_empty_data, result, connect, \
    valued_const_with_data

nodes = lambda output_type: {
    **regular_op_with_empty_data('input', {'type': 'Parameter'}),
    **regular_op_with_empty_data('size', {'op': 'Size', 'type': None, 'output_type': output_type, 'name': 'my_size'}),
    **result(),

    **regular_op_with_empty_data('shape', {'type': 'ShapeOf', 'output_type': output_type}),
    **valued_const_with_data('zero', int64_array([0])),
    **regular_op_with_empty_data('reduce', {'type': 'ReduceProd', 'keep_dims': False}),
}


@generator
class SizeReplacerTest(unittest.TestCase):

    @generate(np.int32, np.int64)
    def test_size_replacer(self, output_type):
        graph = build_graph(nodes_attrs=nodes(output_type), edges=[
            *connect('input', 'size'),
            *connect('size', 'output'),
        ], nodes_with_edges_only=True)
        SizeFrontReplacer().find_and_replace_pattern(graph)

        graph_ref = build_graph(nodes_attrs=nodes(output_type), edges=[
            *connect('input', 'shape'),
            *connect('shape', '0:reduce'),
            *connect('zero', '1:reduce'),
            *connect('reduce', 'output'),
        ], nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.assertEqual(graph.get_op_nodes(type='ReduceProd')[0]['name'], 'my_size',
                         'Name is not inherited from original node for SizeReplacer')
        print(output_type)

    def test_size_replacer_assertion(self):
        graph = build_graph(nodes_attrs=nodes(None), edges=[
            *connect('input', 'size'),
            *connect('size', 'output'),
        ], nodes_with_edges_only=True)
        self.assertRaises(AssertionError, SizeFrontReplacer().find_and_replace_pattern, graph)
