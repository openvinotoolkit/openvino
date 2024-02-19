# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.front.scatter_normalizer import ScatterNormalizer
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, result, connect, \
    regular_op_with_empty_data

nodes = {
    **regular_op_with_empty_data('placeholder_1', {'type': 'Parameter'}),
    **regular_op_with_empty_data('placeholder_2', {'type': 'Parameter'}),
    **regular_op_with_empty_data('placeholder_3', {'type': 'Parameter'}),
    **regular_op_with_empty_data('node', {'op': 'ScatterElementsUpdate', 'is_scatter': True}),
    **regular_op_with_empty_data('axis', {'type': 'Const', 'value': None}),
    **result(),
}

edges = [
    *connect('placeholder_1', '0:node'),
    *connect('placeholder_2', '1:node'),
    *connect('placeholder_3', '2:node'),
    *connect('node', 'output'),
]


class TestDiv(unittest.TestCase):
    def test_ScatterElementsUpdate_has_axis_and_3_inputs(self):
        graph = build_graph(nodes, edges, {'node': {'axis': 1}}, nodes_with_edges_only=True)
        ScatterNormalizer().find_and_replace_pattern(graph)

        graph_ref = build_graph(nodes, [
            *edges,
            *connect('axis', '3:node'),
        ], {'axis': {'value': np.int64(1)}}, nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_ScatterElementsUpdate_has_axis_and_4_inputs(self):
        graph = build_graph(nodes, [
            *edges,
            *connect('axis', '3:node'),
        ], {'node': {'axis': 1}, 'axis': {'value': np.int64(1)}}, nodes_with_edges_only=True)
        self.assertRaises(AssertionError, ScatterNormalizer().find_and_replace_pattern, graph)

    def test_ScatterElementsUpdate_has_no_axis_and_3_inputs(self):
        graph = build_graph(nodes, edges, nodes_with_edges_only=True)
        self.assertRaises(AssertionError, ScatterNormalizer().find_and_replace_pattern, graph)

    def test_ScatterElementsUpdate_has_no_axis_and_4_inputs(self):
        graph = build_graph(nodes, [
            *edges,
            *connect('axis', '3:node'),
        ], {'axis': {'value': np.int64(1)}}, nodes_with_edges_only=True)
        ScatterNormalizer().find_and_replace_pattern(graph)

        graph_ref = build_graph(nodes, [
            *edges,
            *connect('axis', '3:node'),
        ], {'axis': {'value': np.int64(1)}}, nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)
