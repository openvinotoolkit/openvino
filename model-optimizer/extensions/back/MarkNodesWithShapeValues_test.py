# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from extensions.back.MarkNodesWithShapeValues import MarkNodesWithShapeValues
from mo.front.common.partial_infer.utils import int64_array, float32_array
from mo.graph.graph import Node
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph
from mo.utils.unittest.graph import result, regular_op_with_empty_data, \
    shaped_const_with_data, connect, regular_op


class TestMarkDataTypeInShapeOfSubgraphs(unittest.TestCase):

    def test_run_with_shape_subgraph_input(self):
        inp_shape = (1, 3, 1000, 1000)
        dst_type = np.float32

        nodes = {
            **shaped_const_with_data('input', int64_array(inp_shape)),
            **regular_op_with_empty_data('shape', {'type': 'ShapeOf'}),
            **regular_op_with_empty_data('cast_to_float', {'type': 'Cast', 'dst_type': dst_type}),
            **regular_op('mul_const',  {'op': 'Const'}),
            **{'mul_const_d': {'kind': 'data', 'value': float32_array([1., 1., 1., 100.])}},
            **regular_op_with_empty_data('mul', {'type': 'Mul'}),
            **regular_op_with_empty_data('cast_to_int', {'type': 'Cast', 'dst_type': np.int64}),
            **regular_op_with_empty_data('interpolate', {'type': 'Interpolate', 'shape_calculation_model': 'scales'}),
            **result('res'),
        }

        nodes_ref = {
            **shaped_const_with_data('input', int64_array(inp_shape)),
            **regular_op_with_empty_data('shape', {'type': 'ShapeOf'}),
            **regular_op_with_empty_data('cast_to_float', {'type': 'Cast', 'dst_type': dst_type,
                                                           'returns_shape_value': True}),
            **regular_op_with_empty_data('mul', {'type': 'Mul', 'returns_shape_value': True}),
            **regular_op('mul_const',  {'op': 'Const', 'returns_shape_value': True}),
            **{'mul_const_d': {'kind': 'data', 'value': float32_array([1., 1., 1., 100.]),
                               'correct_data_type': True}},
            **regular_op_with_empty_data('cast_to_int', {'type': 'Cast', 'dst_type': np.int64,
                                                         'returns_shape_value': True}),
            **regular_op_with_empty_data('interpolate', {'type': 'Interpolate', 'shape_calculation_model': 'scales'}),
            **result('res'),
        }

        edges = [
            *connect('input', '0:interpolate'),
            *connect('input', '0:shape', skip_data=True),
            *connect('shape', '0:cast_to_float'),
            *connect('cast_to_float', '0:mul'),
            *connect('mul_const', '1:mul'),
            *connect('mul', '0:cast_to_int'),
            *connect('cast_to_int', '1:interpolate'),
            *connect('interpolate', 'res'),
        ]
        graph = build_graph(nodes, edges)
        interp_node = Node(graph, 'interpolate')
        interp_node.add_input_port(2)

        MarkNodesWithShapeValues().find_and_replace_pattern(graph)

        graph_ref = build_graph(nodes_ref, edges)
        (flag, resp) = compare_graphs(graph, graph_ref, 'res', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_run_with_const_input(self):
        inp_shape = (1, 3, 1000, 1000)
        dst_type = np.float32

        nodes = {
            **shaped_const_with_data('input', int64_array(inp_shape)),
            **regular_op('sizes_const',  {'op': 'Const'}),
            **{'sizes_const_d': {'kind': 'data', 'value': float32_array([1., 1., 1., 100.])}},
             **regular_op_with_empty_data('interpolate', {'type': 'Interpolate', 'shape_calculation_model': 'scales'}),
            **result('res'),
        }

        nodes_ref = {
            **shaped_const_with_data('input', int64_array(inp_shape)),
            **regular_op('sizes_const',  {'op': 'Const', 'returns_shape_value': True}),
            **{'sizes_const_d': {'kind': 'data', 'value': float32_array([1., 1., 1., 100.])}},
              **regular_op_with_empty_data('interpolate', {'type': 'Interpolate', 'shape_calculation_model': 'scales'}),
            **result('res'),
        }

        edges = [
            *connect('input', '0:interpolate'),
            *connect('sizes_const', '1:interpolate'),
            *connect('interpolate', 'res'),
        ]
        graph = build_graph(nodes, edges)
        interp_node = Node(graph, 'interpolate')
        interp_node.add_input_port(2)

        MarkNodesWithShapeValues().find_and_replace_pattern(graph)

        graph_ref = build_graph(nodes_ref, edges)
        (flag, resp) = compare_graphs(graph, graph_ref, 'res', check_op_attrs=True)
        self.assertTrue(flag, resp)
