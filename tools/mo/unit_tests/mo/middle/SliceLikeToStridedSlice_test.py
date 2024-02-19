# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.middle.SliceLikeToStridedSlice import SliceLikeToStridedSlice
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph

nodes_attributes = {
    'input': {'kind': 'op', 'op': 'Const'},
    'input_data': {'kind': 'data'},

    'shape_like_input': {'kind': 'op', 'op': 'Const'},
    'shape_like_input_data': {'kind': 'data'},

    'slice_like': {'kind': 'op', 'op': 'slice_like'},
    'slice_like_data': {'kind': 'data', 'shape': None, 'value': None},

    'result': {'kind': 'op', 'op': 'Result'},

    'shape': {'kind': 'op', 'op': 'ShapeOf'},
    'shape_data': {'kind': 'data'},
    'rank_1_d': {'kind': 'op', 'op': 'ShapeOf'},
    'rank_1_d_data': {'kind': 'data'},
    'rank': {'kind': 'op', 'op': 'Squeeze'},
    'rank_data': {'kind': 'data'},
    'rank_const': {'kind': 'op', 'op': 'Const'},
    'rank_const_data': {'kind': 'data'},

    'shape_like': {'kind': 'op', 'op': 'ShapeOf'},
    'shape_like_data': {'kind': 'data'},
    'rank_like_1_d': {'kind': 'op', 'op': 'ShapeOf'},
    'rank_like_1_d_data': {'kind': 'data'},
    'rank_like': {'kind': 'op', 'op': 'Squeeze'},
    'rank_like_const': {'kind': 'op', 'op': 'Const'},
    'rank_like_const_data': {'kind': 'data'},

    'begin': {'kind': 'op', 'op': 'Const'},
    'begin_data': {'kind': 'data'},
    'ss': {'kind': 'op', 'op': 'StridedSlice'},

    'start_idx_like': {'kind': 'op', 'op': 'Const'},
    'start_idx_like_data': {'kind': 'data'},
    'end_idx_like': {'kind': 'op', 'op': 'Const'},
    'end_idx_like_data': {'kind': 'data'},
    'end_idx_like_const': {'kind': 'op', 'op': 'Const'},
    'end_idx_like_const_data': {'kind': 'data'},
    'end_idx_like_add': {'kind': 'op', 'op': 'Add'},
    'end_idx_like_add_data': {'kind': 'data'},
    'delta_like': {'kind': 'op', 'op': 'Const'},
    'delta_like_data': {'kind': 'data'},
    'range_like': {'kind': 'op', 'op': 'Range'},
    'range_like_data': {'kind': 'data'},
    'gather_like': {'kind': 'op', 'op': 't_gather'},
    'gather_like_data': {'kind': 'data'},
    'gather_like_axis': {'kind': 'op', 'op': 'Const'},
    'gather_like_axis_data': {'kind': 'data'},
    'concat': {'kind': 'op', 'op': 'Concat'},
    'concat_data': {'kind': 'data'},

    'start_idx': {'kind': 'op', 'op': 'Const'},
    'start_idx_data': {'kind': 'data'},
    'start_idx_const': {'kind': 'op', 'op': 'Const'},
    'start_idx_const_data': {'kind': 'data'},
    'start_idx_add': {'kind': 'op', 'op': 'Add'},
    'start_idx_add_data': {'kind': 'data'},
    'end_idx': {'kind': 'op', 'op': 'Add'},
    'end_idx_data': {'kind': 'data'},
    'end_idx_axis': {'kind': 'op', 'op': 'Const'},
    'end_idx_axis_data': {'kind': 'data'},
    'end_idx_const': {'kind': 'op', 'op': 'Const'},
    'end_idx_const_data': {'kind': 'data'},
    'end_idx_add': {'kind': 'op', 'op': 'Add'},
    'end_idx_add_data': {'kind': 'data'},
    'delta': {'kind': 'op', 'op': 'Const'},
    'delta_data': {'kind': 'data'},
    'range': {'kind': 'op', 'op': 'Range'},
    'range_data': {'kind': 'data'},
    't_gather': {'kind': 'op', 'op': 't_gather'},
    'gather_data': {'kind': 'data'},
    'gather_axis': {'kind': 'op', 'op': 'Const'},
    'gather_axis_data': {'kind': 'data'}

}

edges = [
    ('input', 'input_data'),
    ('input_data', 'slice_like', {'in': 0}),
    ('shape_like_input', 'shape_like_input_data'),
    ('shape_like_input_data', 'slice_like', {'in': 1}),
    ('slice_like', 'slice_like_data'),
    ('slice_like_data', 'result')
]

same_input_shapes_dims_edges = [
    ('input', 'input_data'),
    ('input_data', 'ss', {'in': 0}),
    ('ss', 'slice_like_data'),
    ('slice_like_data', 'result'),
    ('shape_like_input', 'shape_like_input_data'),
    ('shape_like_input_data', 'shape_like'),
    ('shape_like', 'shape_like_data'),
    ('shape_like_data', 'ss', {'in': 2}),
    ('begin', 'begin_data'),
    ('begin_data', 'ss', {'in': 1})
]

shape_like_sub_graph_edges = [
    ('input', 'input_data'),
    ('input_data', 'ss', {'in': 0}),
    ('ss', 'slice_like_data'),
    ('slice_like_data', 'result'),
    ('begin', 'begin_data'),
    ('begin_data', 'ss', {'in': 1}),
    ('shape_like_input', 'shape_like_input_data'),
    ('shape_like_input_data', 'shape_like'),
    ('shape_like', 'shape_like_data'),
    ('shape_like_data', 'rank_like_1_d'),
    ('rank_like_1_d', 'rank_like_1_d_data'),
    ('rank_like_1_d_data', 'rank_like', {'in': 0}),
    ('rank_like_const', 'rank_like_const_data'),
    ('rank_like_const_data', 'rank_like', {'in': 1}),
    ('end_idx_like', 'end_idx_like_data'),
    ('end_idx_like_const', 'end_idx_like_const_data'),
    ('end_idx_like_data', 'end_idx_like_add', {'in': 0}),
    ('end_idx_like_const_data', 'end_idx_like_add', {'in': 1}),
    ('end_idx_like_add', 'end_idx_like_add_data'),
    ('end_idx_like_add_data', 'range_like', {'in': 1}),
    ('start_idx_like', 'start_idx_like_data'),
    ('start_idx_like_data', 'range_like', {'in': 0}),
    ('delta_like', 'delta_like_data'),
    ('delta_like_data', 'range_like', {'in': 2}),
    ('range_like', 'range_like_data'),
    ('range_like_data', 'gather_like', {'in': 1}),
    ('shape_like_data', 'gather_like', {'in': 0}),
    ('gather_like_axis', 'gather_like_axis_data'),
    ('gather_like_axis_data', 'gather_like', {'in': 2}),
    ('gather_like', 'gather_like_data')
]

last_axis_index = shape_like_sub_graph_edges + [('gather_like_data', 'ss', {'in': 2})]

input_sub_graph_edges = [
    ('input_data', 'shape'),
    ('shape', 'shape_data'),
    ('shape_data', 'rank_1_d'),
    ('rank_1_d', 'rank_1_d_data'),
    ('rank_1_d_data', 'rank', {'in': 0}),
    ('rank_const', 'rank_const_data'),
    ('rank_const_data', 'rank', {'in': 1}),
    ('rank', 'rank_data'),
    ('rank_data', 'end_idx', {'in': 0}),
    ('end_idx_axis', 'end_idx_axis_data'),
    ('end_idx_axis_data', 'end_idx', {'in': 1}),
    ('end_idx', 'end_idx_data'),
    ('end_idx_data', 'end_idx_add', {'in': 0}),
    ('end_idx_const', 'end_idx_const_data'),
    ('end_idx_const_data', 'end_idx_add', {'in': 1}),
    ('start_idx', 'start_idx_data'),
    ('start_idx_data', 'start_idx_add', {'in': 0}),
    ('start_idx_const', 'start_idx_const_data'),
    ('start_idx_const_data', 'start_idx_add', {'in': 1}),
    ('end_idx_add', 'end_idx_add_data'),
    ('start_idx_add', 'start_idx_add_data'),
    ('delta', 'delta_data'),
    ('start_idx_add_data', 'range', {'in': 0}),
    ('end_idx_add_data', 'range', {'in': 1}),
    ('delta_data', 'range', {'in': 2}),
    ('range', 'range_data'),
    ('range_data', 't_gather', {'in': 1}),
    ('shape_data', 't_gather', {'in': 0}),
    ('gather_axis', 'gather_axis_data'),
    ('gather_axis_data', 't_gather', {'in': 2}),
    ('t_gather', 'gather_data'),
    ('gather_data', 'concat', {'in': 1}),
    ('concat', 'concat_data'),
    ('concat_data', 'ss', {'in': 2}),
    ('gather_like_data', 'concat', {'in': 0})
]

input_part_shape_edges = shape_like_sub_graph_edges + input_sub_graph_edges


class SliceLikeToStridedSliceTest(unittest.TestCase):

    def test_1(self):
        graph = build_graph(
            nodes_attributes,
            edges,
            update_attributes={
                'input_data': {'shape': int64_array([1, 224, 224, 3])},
                'shape_like_input_data': {'shape': int64_array([2, 2, 2, 2])},
                'slice_like': {'axes': int64_array([2, 3])}
            },
            nodes_with_edges_only=True
        )
        SliceLikeToStridedSlice().find_and_replace_pattern(graph)
        ref_graph = build_graph(
            nodes_attributes,
            same_input_shapes_dims_edges,
            nodes_with_edges_only=True
        )

        flag, resp = compare_graphs(graph, ref_graph, 'result')
        self.assertTrue(flag, resp)

    def test_2(self):
        graph = build_graph(
            nodes_attributes,
            edges,
            update_attributes={
                'input_data': {'shape': int64_array([1, 224, 224, 3])},
                'shape_like_input_data': {'shape': int64_array([2, 2, 2, 2, 2])},
                'slice_like': {'axes': int64_array([2, 3])}
            },
            nodes_with_edges_only=True
        )
        SliceLikeToStridedSlice().find_and_replace_pattern(graph)
        ref_graph = build_graph(
            nodes_attributes,
            last_axis_index,
            nodes_with_edges_only=True
        )

        flag, resp = compare_graphs(graph, ref_graph, 'result')
        self.assertTrue(flag, resp)

    def test_3(self):
        graph = build_graph(
            nodes_attributes,
            edges,
            update_attributes={
                'input_data': {'shape': int64_array([1, 224, 224, 3])},
                'shape_like_input_data': {'shape': int64_array([2, 2, 2, 2, 2])},
                'slice_like': {'axes': int64_array([1, 2])}
            },
            nodes_with_edges_only=True
        )
        SliceLikeToStridedSlice().find_and_replace_pattern(graph)
        ref_graph = build_graph(
            nodes_attributes,
            input_part_shape_edges,
            nodes_with_edges_only=True
        )
        flag, resp = compare_graphs(graph, ref_graph, 'result')
        self.assertTrue(flag, resp)
