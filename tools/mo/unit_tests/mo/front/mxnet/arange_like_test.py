# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import unittest
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.mxnet.arange_like_replacer import ArangeLikeReplacer
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, shaped_parameter, regular_op_with_empty_data, result, connect, \
    shaped_const_with_data, connect_data


class ArangeLikeReplacerTest(unittest.TestCase):
    def test_axis_not_none_start_0(self):
        graph = build_graph(
            nodes_attrs={
                **shaped_parameter('input', int64_array([1, 3, 5, 5])),
                **regular_op_with_empty_data('arange_like', {'op': 'arange_like', 'type': None, 'axis': 3, 'repeat': 1,
                                                             'start': 0, 'step': 1}),
                **result('result')
            },
            edges=[
                *connect('input', 'arange_like'),
                *connect('arange_like', 'result')
            ]
        )
        ref_graph = build_graph(
            nodes_attrs={
                **shaped_parameter('input', int64_array([1, 3, 5, 5])),
                **regular_op_with_empty_data('shape_of', {'op': 'ShapeOf', 'type': 'ShapeOf'}),
                **shaped_const_with_data('gather_axis', None),
                **shaped_const_with_data('gather_indices', None),
                **regular_op_with_empty_data('gather', {'op': 'Gather', 'type': 'Gather'}),
                **shaped_const_with_data('range_start', None),
                **shaped_const_with_data('range_step', None),
                **shaped_const_with_data('squeeze_const', None),
                **regular_op_with_empty_data('squeeze', {'op': 'Squeeze', 'type': 'Squeeze'}),
                **regular_op_with_empty_data('range', {'op': 'Range', 'type': 'Range'}),
                **result('result')
            },
            edges=[
                *connect('input', 'shape_of'),
                *connect('shape_of', '0:gather'),
                *connect('gather_axis', '1:gather'),
                *connect('gather_indices', '2:gather'),
                *connect('range_start', '0:range'),
                *connect('gather', '0:squeeze'),
                *connect('squeeze_const', '1:squeeze'),
                *connect('squeeze', '1:range'),
                *connect('range_step', '2:range'),
                *connect('range', 'result')
            ],
            update_attributes={
                'gather_axis': {'value': 3},
                'gather_indices': {'value': 0},
                'range_start': {'value': 0},
                'range_step': {'value': 1}
            }
        )
        ArangeLikeReplacer().find_and_replace_pattern(graph)
        flag, resp = compare_graphs(graph, ref_graph, 'result', 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_axis_not_none_start_1_step_2(self):
        graph = build_graph(
            nodes_attrs={
                **shaped_parameter('input', int64_array([1, 3, 5, 5])),
                **regular_op_with_empty_data('arange_like', {'op': 'arange_like', 'type': None, 'axis': 3, 'repeat': 1,
                                                             'start': 1, 'step': 2}),
                **result('result')
            },
            edges=[
                *connect('input', 'arange_like'),
                *connect('arange_like', 'result')
            ]
        )
        ref_graph = build_graph(
            nodes_attrs={
                **shaped_parameter('input', int64_array([1, 3, 5, 5])),
                **regular_op_with_empty_data('shape_of', {'op': 'ShapeOf', 'type': 'ShapeOf'}),
                **shaped_const_with_data('gather_axis', None),
                **shaped_const_with_data('gather_indices', None),
                **regular_op_with_empty_data('gather', {'op': 'Gather', 'type': 'Gather'}),
                **regular_op_with_empty_data('mul', {'op': 'Mul', 'type': 'Multiply'}),
                **shaped_const_with_data('mul_const', None),
                **shaped_const_with_data('range_start', None),
                **shaped_const_with_data('range_step', None),
                **shaped_const_with_data('add_const', None),
                **regular_op_with_empty_data('add', {'op': 'Add', 'type': 'Add'}),
                **shaped_const_with_data('squeeze_const', None),
                **regular_op_with_empty_data('squeeze', {'op': 'Squeeze', 'type': 'Squeeze'}),
                **regular_op_with_empty_data('range', {'op': 'Range', 'type': 'Range'}),
                **regular_op_with_empty_data('slice', {'op': 'Slice', 'type': None}),
                **shaped_const_with_data('slice_start', None),
                **shaped_const_with_data('slice_axes', None),
                **shaped_const_with_data('slice_step', None),
                **result('result')
            },
            edges=[
                *connect('input', 'shape_of'),
                *connect('shape_of', '0:gather'),
                *connect('gather_axis', '1:gather'),
                *connect('gather_indices', '2:gather'),
                *connect('range_start', '0:range'),
                *connect('gather', '0:mul'),
                *connect('mul_const', '1:mul'),
                *connect('mul', '0:add'),
                *connect('add_const', '1:add'),
                *connect('squeeze_const', '1:squeeze'),
                *connect('add', '0:squeeze'),
                *connect('squeeze', '1:range'),
                *connect('range_step', '2:range'),
                *connect('range', '0:slice'),
                *connect('slice_start', '1:slice'),
                *connect_data('gather', '2:slice'),
                *connect('slice_axes', '3:slice'),
                *connect('slice_step', '4:slice'),
                *connect('slice', 'result')
            ],
            update_attributes={
                'gather_axis': {'value': 3},
                'gather_indices': {'value': 0},
                'range_start': {'value': 1},
                'range_step': {'value': 2},
                'add_const': {'value': 1},
                'mul_const': {'value': 2},
                'slice_start': {'value': int64_array([0])},
                'slice_axes': {'value': int64_array([0])},
                'slice_step': {'value': int64_array([1])},
            }
        )
        ArangeLikeReplacer().find_and_replace_pattern(graph)
        flag, resp = compare_graphs(graph, ref_graph, 'result', 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_axis_none_start_0(self):
        graph = build_graph(
            nodes_attrs={
                **shaped_parameter('input', int64_array([1, 3, 5, 5])),
                **regular_op_with_empty_data('arange_like', {'op': 'arange_like', 'type': None, 'axis': None,
                                                             'repeat': 1, 'start': 0, 'step': 1}),
                **result('result')
            },
            edges=[
                *connect('input', 'arange_like'),
                *connect('arange_like', 'result')
            ]
        )
        ref_graph = build_graph(
            nodes_attrs={
                **shaped_parameter('input', int64_array([1, 3, 5, 5])),
                **regular_op_with_empty_data('shape_of', {'op': 'ShapeOf', 'type': 'ShapeOf'}),
                **regular_op_with_empty_data('reduce_prod', {'op': 'ReduceProd', 'type': 'ReduceProd'}),
                **shaped_const_with_data('reduce_prod_const', None),
                **shaped_const_with_data('squeeze_const', None),
                **regular_op_with_empty_data('squeeze', {'op': 'Squeeze', 'type': 'Squeeze'}),
                **shaped_const_with_data('range_start', None),
                **shaped_const_with_data('range_step', None),
                **regular_op_with_empty_data('range', {'op': 'Range', 'type': 'Range'}),
                **regular_op_with_empty_data('reshape_backward', {'op': 'Reshape', 'type': 'Reshape'}),
                **result('result')
            },
            edges=[
                *connect('input', 'shape_of'),
                *connect('shape_of', '0:reduce_prod'),
                *connect('reduce_prod_const', '1:reduce_prod'),
                *connect('squeeze_const', '1:squeeze'),
                *connect('reduce_prod', '0:squeeze'),
                *connect('range_start', '0:range'),
                *connect('range_step', '2:range'),
                *connect('squeeze', '1:range'),
                *connect('range', '0:reshape_backward'),
                *connect_data('shape_of', '1:reshape_backward'),
                *connect('reshape_backward', 'result')
            ],
            update_attributes={
                'range_start': {'value': 0},
                'range_step': {'value': 1},
                'reduce_prod_const': {'value': int64_array([0])}
            }
        )

        ArangeLikeReplacer().find_and_replace_pattern(graph)
        flag, resp = compare_graphs(graph, ref_graph, 'result', 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_axis_none_start_1(self):
        graph = build_graph(
            nodes_attrs={
                **shaped_parameter('input', int64_array([1, 3, 5, 5])),
                **regular_op_with_empty_data('arange_like', {'op': 'arange_like', 'type': None, 'axis': None,
                                                             'repeat': 1, 'start': 1, 'step': 1}),
                **result('result')
            },
            edges=[
                *connect('input', 'arange_like'),
                *connect('arange_like', 'result')
            ]
        )
        ref_graph = build_graph(
            nodes_attrs={
                **shaped_parameter('input', int64_array([1, 3, 5, 5])),
                **regular_op_with_empty_data('shape_of', {'op': 'ShapeOf', 'type': 'ShapeOf'}),
                **regular_op_with_empty_data('reduce_prod', {'op': 'ReduceProd', 'type': 'ReduceProd'}),
                **shaped_const_with_data('reduce_prod_const', None),
                **shaped_const_with_data('squeeze_const', None),
                **regular_op_with_empty_data('squeeze', {'op': 'Squeeze', 'type': 'Squeeze'}),
                **shaped_const_with_data('add_const', None),
                **regular_op_with_empty_data('add', {'op': 'Add', 'type': 'Add'}),
                **shaped_const_with_data('range_start', None),
                **shaped_const_with_data('range_step', None),
                **regular_op_with_empty_data('range', {'op': 'Range', 'type': 'Range'}),
                **regular_op_with_empty_data('reshape_backward', {'op': 'Reshape', 'type': 'Reshape'}),
                **result('result')
            },
            edges=[
                *connect('input', 'shape_of'),
                *connect('shape_of', '0:reduce_prod'),
                *connect('reduce_prod_const', '1:reduce_prod'),
                *connect('squeeze_const', '1:squeeze'),
                *connect('add_const', '1:add'),
                *connect('reduce_prod', '0:add'),
                *connect('add', '0:squeeze'),
                *connect('range_start', '0:range'),
                *connect('range_step', '2:range'),
                *connect('squeeze', '1:range'),
                *connect('range', '0:reshape_backward'),
                *connect_data('shape_of', '1:reshape_backward'),
                *connect('reshape_backward', 'result')
            ],
            update_attributes={
                'range_start': {'value': 1},
                'range_step': {'value': 1},
                'add_const': {'value': 1},
                'reduce_prod_const': {'value': int64_array([0])}
            }
        )
        ArangeLikeReplacer().find_and_replace_pattern(graph)
        flag, resp = compare_graphs(graph, ref_graph, 'result', 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)
