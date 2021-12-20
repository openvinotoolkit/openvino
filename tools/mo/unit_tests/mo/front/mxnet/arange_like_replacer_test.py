# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.mxnet.arange_like_replacer import ArangeLikeReplacer
from openvino.tools.mo.utils.error import Error
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

    def test_axis_not_none_start_1(self):
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
                **shaped_const_with_data('range_start', None),
                **shaped_const_with_data('range_step', None),
                **shaped_const_with_data('add_const', None),
                **regular_op_with_empty_data('add', {'op': 'Add', 'type': 'Add'}),
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
                *connect('gather', '0:add'),
                *connect('add_const', '1:add'),
                *connect('squeeze_const', '1:squeeze'),
                *connect('add', '0:squeeze'),
                *connect('squeeze', '1:range'),
                *connect('range_step', '2:range'),
                *connect('range', 'result')
            ],
            update_attributes={
                'gather_axis': {'value': 3},
                'gather_indices': {'value': 0},
                'range_start': {'value': 1},
                'range_step': {'value': 2},
                'add_const': {'value': 1}
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
                                                             'repeat': 1, 'start': 0, 'step': 2}),
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
                **shaped_const_with_data('reshape_forward_const', None),
                **regular_op_with_empty_data('reshape_forward', {'op': 'Reshape', 'type': 'Reshape'}),
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
                *connect('shape_of', '0:reshape_forward'),
                *connect('reshape_forward_const', '1:reshape_forward'),
                *connect('squeeze_const', '1:squeeze'),
                *connect('reshape_forward', '0:squeeze'),
                *connect('range_start', '0:range'),
                *connect('range_step', '2:range'),
                *connect('squeeze', '1:range'),
                *connect('range', '0:reshape_backward'),
                *connect_data('shape_of', '1:reshape_backward'),
                *connect('reshape_backward', 'result')
            ],
            update_attributes={
                'range_start': {'value': 0},
                'range_step': {'value': 2}
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
                **shaped_const_with_data('reshape_forward_const', None),
                **regular_op_with_empty_data('reshape_forward', {'op': 'Reshape', 'type': 'Reshape'}),
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
                *connect('shape_of', '0:reshape_forward'),
                *connect('reshape_forward_const', '1:reshape_forward'),
                *connect('squeeze_const', '1:squeeze'),
                *connect('add_const', '1:add'),
                *connect('reshape_forward', '0:add'),
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
                'add_const': {'value': 1}
            }
        )

        ArangeLikeReplacer().find_and_replace_pattern(graph)
        flag, resp = compare_graphs(graph, ref_graph, 'result', 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_negative_repeat_non_default(self):
        graph = build_graph(
            nodes_attrs={
                **shaped_parameter('input', int64_array([1, 3, 5, 5])),
                **regular_op_with_empty_data('arange_like', {'op': 'arange_like', 'type': None, 'axis': None,
                                                             'repeat': 2, 'start': 1, 'step': 1}),
                **result('result')
            },
            edges=[
                *connect('input', 'arange_like'),
                *connect('arange_like', 'result')
            ]
        )

        self.assertRaises(Error, ArangeLikeReplacer().find_and_replace_pattern, graph)
