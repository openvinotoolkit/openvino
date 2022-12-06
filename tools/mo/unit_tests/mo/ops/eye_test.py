# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
import numpy as np

from generator import generator, generate

from openvino.tools.mo.ops.eye import Eye
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.graph.graph import Node
from unit_tests.utils.graph import build_graph_with_attrs
from openvino.tools.mo.front.common.partial_infer.utils import int64_array, shape_array, dynamic_dimension_value


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
