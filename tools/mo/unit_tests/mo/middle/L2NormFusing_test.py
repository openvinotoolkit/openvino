# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

import numpy as np

from openvino.tools.mo.middle.L2NormFusing import L2NormToNorm
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph_with_attrs

# A list with nodes attributes used to build various graphs.
nodes = [
    ('l2_normalize_mul', dict(kind='op', op='Mul', name='l2_norm_name')),
    ('l2_normalize_mul_data', dict(kind='data')),
    ('maximum', dict(kind='op', op='Maximum')),
    ('maximum_data', dict(kind='data')),
    ('maximum_y_const', dict(kind='op', op='Const', value=np.array(12.e-13, dtype=np.float32))),
    ('maximum_y_data', dict(kind='data', value=np.array(12.e-13, dtype=np.float32))),
    ('rsqrt_pow', dict(kind='data', value=-0.5)),
    ('rsqrt', dict(kind='op', op='Pow')),
    ('rsqrt_data', dict(kind='data')),
    ('square_pow', dict(kind='op', op='Const', value=2.)),
    ('square_pow_data', dict(kind='data', value=2.)),
    ('square', dict(kind='op', op='Pow')),
    ('sum', dict(kind='op', op='ReduceSum')),
    ('sum_data', dict(kind='data')),
    ('sum_axes', dict(kind='op', op='Const')),
    # nodes added after replacement
    ('normalize_node', dict(kind='op', op='NormalizeL2')),
    ('weights_node', dict(kind='op', op='Const')),
    ('result', dict(kind='op', op='Result'))
]

edges = [
    ('input', 'input_data', {'out': 0}),
    ('input_data', 'square', {'in': 0}),
    ('square_pow', 'square_pow_data', {'out': 0}),
    ('square_pow_data', 'square', {'in': 1}),
    ('square', 'square_data'),
    ('square_data', 'sum'),
    ('sum_axes', 'sum_axes_data'),
    ('sum_axes_data', 'sum'),
    ('sum', 'sum_data'),
    ('maximum_y_const', 'maximum_y_data'),
    ('maximum_y_data', 'maximum'),
    ('sum_data', 'maximum'),
    ('maximum', 'maximum_data'),
    ('maximum_data', 'rsqrt', {'in': 0}),
    ('rsqrt_pow', 'rsqrt', {'in': 1}),
    ('rsqrt', 'rsqrt_data'),
    ('rsqrt_data', 'l2_normalize_mul'),
    ('input_data', 'l2_normalize_mul'),
    ('l2_normalize_mul', 'l2_normalize_mul_data'),
    ('l2_normalize_mul_data', 'result'),
]

edges_after_replacement = [
    ('input', 'input_data', {'out': 0}),
    ('input_data', 'normalize_node'),
    ('weights_node', 'weights_node_data'),
    ('weights_node_data', 'normalize_node'),
    ('normalize_node', 'l2_normalize_mul_data'),
    ('l2_normalize_mul_data', 'result'),
]


class TestL2NormToNormTest():
    @pytest.mark.parametrize("input_shape, axes, layout",
                             [(int64_array([2, 3]), int64_array([1]), 'NCHW'),  # NC layout, normalize C dimension
                (int64_array([2, 3]), int64_array([1]), 'NHWC'),  # NC layout, normalize C dimension
                (int64_array([2, 3, 5]), int64_array([1]), 'NCHW'),  # NCH layout, normalize C dimension
                (int64_array([2, 3, 5]), int64_array([1]), 'NHWC'),  # NCH layout, normalize C dimension
                (int64_array([2, 3, 5]), int64_array([-1, -2]), 'NHWC'),  # NCH layout, normalize CH dimensions
                (int64_array([2, 3, 5]), int64_array([-1, -2]), 'NCHW'),  # NCH layout, normalize CH dimensions
                (int64_array([2, 3, 5]), int64_array([1, 2]), 'NCHW'),  # NCH layout, normalize CH dimensions
                (int64_array([2, 3, 5]), int64_array([1, 2]), 'NHWC'),  # NCH layout, normalize CH dimensions
                (int64_array([2, 3, 5, 7]), int64_array([1]), 'NCHW'),  # NCHW layout, normalize C dimension
                (int64_array([2, 3, 5, 7]), int64_array([-1]), 'NHWC'),  # NHWC layout, normalize C dimension
                (int64_array([2, 3, 5, 7]), int64_array([3]), 'NHWC'),  # NCHW layout, normalize C dimension
                (int64_array([2, 3, 5, 7]), int64_array([-1, 1, 2]), 'NCHW'),  # NCHW layout, normalize CHW dimensions
                (int64_array([2, 3, 5, 7]), int64_array([-3, -2, -1]), 'NHWC'),  # NCHW layout, normalize HWC dimensions
                ])
    def test_positive(self, input_shape, axes, layout):
        graph = build_graph_with_attrs(nodes + [
            ('input', dict(kind='op', shape=input_shape, op='Parameter', data_type=np.float32)),
            ('input_data', dict(kind='data', shape=input_shape, data_type=np.float32)),
            ('square_data', dict(kind='data', shape=input_shape)),
            ('sum_axes_data', dict(kind='data', value=axes, shape=None)),
        ], edges, nodes_with_edges_only=True)
        graph.stage = 'middle'
        graph.graph['layout'] = layout

        L2NormToNorm().find_and_replace_pattern(graph)

        graph_ref = build_graph_with_attrs(nodes + [
            ('input', dict(kind='op', shape=input_shape, op='Parameter', data_type=np.float32)),
            ('input_data', dict(kind='data', shape=input_shape, data_type=np.float32)),
            ('weights_node_data', dict(kind='data', value=axes.sort())),
        ], edges_after_replacement, nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        assert (graph.node[graph.get_nodes_with_attributes(type='NormalizeL2')[0]]['name'] == 'l2_norm_name')
        assert flag, resp

    @pytest.mark.parametrize("input_shape, axes, layout",
                             [(int64_array([2]), int64_array([0]), 'NCHW'),
                (int64_array([2, 3]), int64_array([0]), 'NCHW'),
                (int64_array([2, 3]), int64_array([0]), 'NHWC'),
                (int64_array([2, 3]), int64_array([0, 1]), 'NCHW'),
                (int64_array([2, 3]), int64_array([0, 1]), 'NHWC'),
                (int64_array([2, 3, 5]), int64_array([0]), 'NCHW'),
                (int64_array([2, 3, 5]), int64_array([0]), 'NHWC'),
                (int64_array([2, 3, 5]), int64_array([-1]), 'NCHW'),
                (int64_array([2, 3, 5]), int64_array([-1]), 'NHWC'),
                (int64_array([2, 3, 5]), int64_array([0, 1]), 'NCHW'),
                (int64_array([2, 3, 5]), int64_array([0, 1]), 'NHWC'),
                (int64_array([2, 3, 5]), int64_array([0, 2]), 'NCHW'),
                (int64_array([2, 3, 5]), int64_array([0, 2]), 'NHWC'),
                (int64_array([2, 3, 5, 7]), int64_array([0]), 'NCHW'),
                (int64_array([2, 3, 5, 7]), int64_array([0]), 'NHWC'),
                (int64_array([2, 3, 5, 7]), int64_array([2]), 'NCHW'),
                (int64_array([2, 3, 5, 7]), int64_array([2]), 'NHWC'),
                (int64_array([2, 3, 5, 7]), int64_array([3]), 'NCHW'),
                (int64_array([2, 3, 5, 7]), int64_array([1]), 'NHWC'),
                (int64_array([2, 3, 5, 7]), int64_array([1, 2]), 'NCHW'),
                (int64_array([2, 3, 5, 7]), int64_array([1, -1]), 'NHWC'),
                (int64_array([2, 3, 5, 7]), int64_array([1, -1]), 'NCHW'),
                (int64_array([2, 3, 5, 7]), int64_array([-2, -1]), 'NHWC'),
                (int64_array([2, 3, 5, 7]), int64_array([1, 3]), 'NCHW'),
                (int64_array([2, 3, 5, 7]), int64_array([2, 3]), 'NHWC'),
                (int64_array([2, 3, 5, 7]), int64_array([0, 1, 2]), 'NCHW'),
                (int64_array([2, 3, 5, 7]), int64_array([0, 1, 2]), 'NHWC'),
                (int64_array([2, 3, 5, 7]), int64_array([0, 2, 3]), 'NCHW'),
                (int64_array([2, 3, 5, 7]), int64_array([0, 2, 3]), 'NHWC'),
                (int64_array([2, 3, 5, 7]), int64_array([0, 1, 2, 3]), 'NCHW'),
                (int64_array([2, 3, 5, 7]), int64_array([0, 1, 2, 3]), 'NHWC'),
                (int64_array([2, 3, 5, 7, 9]), int64_array([1]), 'NCHW'),
                (int64_array([2, 3, 5, 7, 9]), int64_array([-1]), 'NHWC'),
                (int64_array([2, 3, 5, 7, 9]), int64_array([1, 2, 3, 4]), 'NCHW'),
                (int64_array([2, 3, 5, 7, 9]), int64_array([-1, -2, -3, -4]), 'NHWC'),
                ])
    def test_negative(self, input_shape, axes, layout):
        graph = build_graph_with_attrs(nodes + [
            ('input', dict(kind='op', shape=input_shape, op='Parameter', data_type=np.float32)),
            ('input_data', dict(kind='data', shape=input_shape, data_type=np.float32)),
            ('square_data', dict(kind='data', shape=input_shape)),
            ('sum_axes_data', dict(kind='data', value=axes, shape=None)),
        ], edges, nodes_with_edges_only=True)
        graph.stage = 'middle'
        graph.graph['layout'] = layout

        L2NormToNorm().find_and_replace_pattern(graph)

        graph_ref = build_graph_with_attrs(nodes + [
            ('input', dict(kind='op', shape=input_shape, op='Parameter', data_type=np.float32)),
            ('input_data', dict(kind='data', shape=input_shape, data_type=np.float32)),
            ('square_data', dict(kind='data', shape=input_shape)),
            ('sum_axes_data', dict(kind='data', value=axes, shape=None)),
        ], edges, nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        assert flag, resp
