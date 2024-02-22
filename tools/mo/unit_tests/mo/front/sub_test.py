# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.front.sub import Sub
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, result, regular_op_with_shaped_data, valued_const_with_data, connect, \
    connect_data

nodes = {
    **regular_op_with_shaped_data('placeholder_1', [1, 227, 227, 3], {'type': 'Parameter'}),
    **regular_op_with_shaped_data('placeholder_2', [1, 227, 227, 3], {'type': 'Parameter'}),
    **regular_op_with_shaped_data('sub', None, {'op': 'Sub', 'type': 'Subtract', 'name': 'my_sub'}),

    **regular_op_with_shaped_data('negate', [1, 227, 227, 3], {'type': 'Multiply'}),
    **valued_const_with_data('minus_one', np.array(-1.)),
    **regular_op_with_shaped_data('add', None, {'type': 'Add'}),

    **result(),
}


class TestSub(unittest.TestCase):
    def test_sub_test_1(self):
        # Test with two different inputs from two placeholders
        graph = build_graph(nodes, [
            *connect('placeholder_1', '0:sub'),
            *connect('placeholder_2', '1:sub'),
            *connect('sub', 'output'),
        ], nodes_with_edges_only=True)
        Sub().find_and_replace_pattern(graph)

        graph_ref = build_graph(nodes, [
            *connect('placeholder_1', '0:add'),
            *connect('placeholder_2', '0:negate'),
            *connect('minus_one', '1:negate'),
            *connect('negate', '1:add'),
            *connect('add', 'output'),
        ], nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.assertTrue(graph.node[graph.get_nodes_with_attributes(type='Add')[0]]['name'] == 'my_sub')

    def test_sub_test_2(self):
        # Test with two same inputs from one placeholder
        graph = build_graph(nodes, [
            *connect('placeholder_1:0', '0:sub'),
            *connect_data('placeholder_1:0', '1:sub'),
            *connect('sub', 'output'),
        ], nodes_with_edges_only=True)
        Sub().find_and_replace_pattern(graph)

        graph_ref = build_graph(nodes, [
            *connect('placeholder_1:0', '0:add'),
            *connect_data('placeholder_1:0', '0:negate'),
            *connect('minus_one', '1:negate'),
            *connect('negate', '1:add'),
            *connect('add', 'output'),
        ], nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.assertTrue(graph.node[graph.get_nodes_with_attributes(type='Add')[0]]['name'] == 'my_sub')
