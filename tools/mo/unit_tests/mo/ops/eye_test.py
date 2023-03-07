# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
import numpy as np

from generator import generator, generate

from openvino.tools.mo.ops.eye import Eye
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.graph.graph import Node
from unit_tests.utils.graph import build_graph_with_attrs, build_graph
from openvino.tools.mo.front.common.partial_infer.utils import int64_array, dynamic_dimension_value
from unit_tests.utils.graph import valued_const_with_data, result, regular_op_with_empty_data, connect


graph_node_attrs_sizes = [
    ('num_rows', {'type': 'Parameter', 'kind': 'op'}),
    ('num_rows_data', {'kind': 'data'}),

    ('num_columns', {'type': 'Parameter', 'kind': 'op'}),
    ('num_columns_data', {'kind': 'data'}),

    ('diagonal_index', {'type': 'Parameter', 'kind': 'op'}),
    ('diagonal_index_data', {'kind': 'data'}),

    ('batch_shape', {'type': 'Parameter', 'kind': 'op'}),
    ('batch_shape_data', {'kind': 'data'}),

    ('eye_op', {'type': 'Eye', 'kind': 'op'}),
    ('eye_op_data', {'kind': 'data', 'shape': None, 'value': None}),

    ('op_output', {'kind': 'op', 'op': 'Result'}),
]


graph_edges_sizes = [
    ('num_rows', 'num_rows_data'),
    ('num_columns', 'num_columns_data'),
    ('diagonal_index', 'diagonal_index_data'),
    ('batch_shape', 'batch_shape_data'),

    ('num_rows_data', 'eye_op', {'in': 0}),
    ('num_columns_data', 'eye_op', {'in': 1}),
    ('diagonal_index_data', 'eye_op', {'in': 2}),
    ('batch_shape_data', 'eye_op', {'in': 3}),

    ('eye_op', 'eye_op_data'),
    ('eye_op_data', 'op_output'),
]


@generator
class TestComplexOp(unittest.TestCase):
    @generate(*[
        ([], [dynamic_dimension_value, dynamic_dimension_value]),
        ([1], [dynamic_dimension_value, dynamic_dimension_value]),
        ([1], [2, dynamic_dimension_value, dynamic_dimension_value], None, None, [2]),
        ([1], [2, 3, dynamic_dimension_value], 3, None, [2]),
        ([1], [2, dynamic_dimension_value, 4], None, 4, [2]),
        ([1], [2, 3, 4], [3], [4], [2])
    ])
    def test_complex_op_shape_inference(self, input_shape, output_shape, num_rows=None, num_cols=None, batch_shape=[]):
        graph = build_graph_with_attrs(nodes_with_attrs=graph_node_attrs_sizes,
                                       edges_with_attrs=graph_edges_sizes,
                                       update_nodes_attributes=[
                                           ('num_rows_data', {'shape': int64_array(input_shape), 'value': num_rows}),
                                           ('num_columns_data', {'shape': int64_array(input_shape), 'value': num_cols}),
                                           ('diagonal_index_data', {'shape': int64_array(input_shape)}),
                                           ('batch_shape_data', {'shape': int64_array([len(batch_shape)]), 'value': batch_shape}),
                                           ('eye_op', {'output_type': np.float32}),
                                       ])
        node = Node(graph, 'eye_op')
        Eye.infer(node)

        msg = "Eye operation infer failed for case: expected_shape={}, actual_shape={}"

        self.assertTrue(np.array_equal(graph.node['eye_op_data']['shape'], output_shape),
                        msg.format(output_shape, graph.node['eye_op_data']['shape']))

    def test_value_inference(self):
        graph_node_attrs_sizes = {
            **valued_const_with_data('num_rows', int64_array([128])),
            **valued_const_with_data('num_columns', int64_array([128])),
            **valued_const_with_data('diagonal_index', int64_array([0])),
            **valued_const_with_data('batch_shape', int64_array([])),
            **regular_op_with_empty_data('eye_op', {'op': 'Eye', 'output_type': np.float32}),
            **result('res'),
        }
        graph_edges_sizes = [
            *connect('num_rows', '0:eye_op'),
            *connect('num_columns', '1:eye_op'),
            *connect('diagonal_index', '2:eye_op'),
            *connect('batch_shape', '3:eye_op'),
            *connect('eye_op', 'res'),
        ]
        graph = build_graph(
            graph_node_attrs_sizes, graph_edges_sizes)
        node = Node(graph, 'eye_op')
        Eye.infer(node)
        output_value = np.eye(int64_array(128), M=int64_array(
            128), k=int64_array(0), dtype=np.float32)

        msg = "Eye operation infer failed for case: expected_value={}, actual_value={}"

        self.assertTrue(np.array_equal(graph.node['eye_op_d']['value'], output_value),
                        msg.format(output_value, graph.node['eye_op_d']['value']))
