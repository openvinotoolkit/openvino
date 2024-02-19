# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.middle.SliceConverter import ConvertSlice
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, regular_op_with_shaped_data, valued_const_with_data, \
    regular_op_with_empty_data, result, connect, connect_data

nodes_attributes = {
    **regular_op_with_shaped_data('input', [2, 3, 300, 300], {'type': 'Parameter', 'op': 'Parameter'}),
    **regular_op_with_empty_data('starts', {'op': 'Const', 'type': 'Const'}),
    **regular_op_with_empty_data('ends', {'op': 'Const', 'type': 'Const'}),
    **regular_op_with_empty_data('axes', {'op': 'Const', 'type': 'Const'}),
    **regular_op_with_empty_data('steps', {'op': 'Const', 'type': 'Const'}),
    **regular_op_with_empty_data('slice', {'op': 'Slice', 'type': None}),

    **regular_op_with_empty_data('ss_begin_cast', {'op': 'Cast', 'type': 'Convert', 'dst_type': np.int64}),
    **regular_op_with_empty_data('ss_begin_clamp', {'op': 'Clamp', 'type': None}),
    **regular_op_with_empty_data('ss_begin_clamp_min', {'value': np.iinfo(np.int32).min, 'op': 'Const', 'type': 'Const'}),
    **regular_op_with_empty_data('ss_begin_clamp_max', {'value': np.iinfo(np.int32).max, 'op': 'Const', 'type': 'Const'}),
    **regular_op_with_empty_data('ss_begin_gather_0', {'op': 'Gather', 'type': 'Gather'}),
    **valued_const_with_data('ss_begin_gather_0_idx', int64_array([0])),
    **regular_op_with_shaped_data('ss_begin_gather_0_axis', [], {'op': 'Const', 'type': 'Const', 'value': [0]}),
    **regular_op_with_empty_data('ss_begin_gather_1', {'op': 'Gather', 'type': 'Gather'}),
    **valued_const_with_data('ss_begin_gather_1_idx', int64_array([1])),
    **regular_op_with_shaped_data('ss_begin_gather_1_axis', [], {'op': 'Const', 'type': 'Const', 'value': [0]}),
    **regular_op_with_empty_data('ss_begin_gather_2', {'op': 'Gather', 'type': 'Gather'}),
    **valued_const_with_data('ss_begin_gather_2_idx', int64_array([2])),
    **regular_op_with_shaped_data('ss_begin_gather_2_axis', [], {'op': 'Const', 'type': 'Const', 'value': [0]}),
    **regular_op_with_empty_data('ss_begin_gather_3', {'op': 'Gather', 'type': 'Gather'}),
    **valued_const_with_data('ss_begin_gather_3_idx', int64_array([3])),
    **regular_op_with_shaped_data('ss_begin_gather_3_axis', [], {'op': 'Const', 'type': 'Const', 'value': [0]}),
    **regular_op_with_empty_data('ss_begin_const_0', {'op': 'Const', 'type': 'Const', 'value': int64_array([0])}),
    **regular_op_with_empty_data('ss_begin_const_1', {'op': 'Const', 'type': 'Const', 'value': int64_array([0])}),
    **regular_op_with_empty_data('ss_begin_const_2', {'op': 'Const', 'type': 'Const', 'value': int64_array([0])}),
    **regular_op_with_empty_data('ss_begin_const_3', {'op': 'Const', 'type': 'Const', 'value': int64_array([0])}),
    **regular_op_with_empty_data('ss_begin_concat', {'op': 'Concat', 'type': 'Concat'}),

    **regular_op_with_empty_data('ss_end_cast', {'op': 'Cast', 'type': 'Convert', 'dst_type': np.int64}),
    **regular_op_with_empty_data('ss_end_clamp', {'op': 'Clamp', 'type': None}),
    **regular_op_with_empty_data('ss_end_clamp_min', {'value': np.iinfo(np.int32).min, 'op': 'Const', 'type': 'Const'}),
    **regular_op_with_empty_data('ss_end_clamp_max', {'value': np.iinfo(np.int32).max, 'op': 'Const', 'type': 'Const'}),
    **regular_op_with_empty_data('ss_end_gather_0', {'op': 'Gather', 'type': 'Gather'}),
    **valued_const_with_data('ss_end_gather_0_idx', int64_array([0])),
    **regular_op_with_shaped_data('ss_end_gather_0_axis', [], {'op': 'Const', 'type': 'Const', 'value': [0]}),
    **regular_op_with_empty_data('ss_end_gather_1', {'op': 'Gather', 'type': 'Gather'}),
    **valued_const_with_data('ss_end_gather_1_idx', int64_array([1])),
    **regular_op_with_shaped_data('ss_end_gather_1_axis', [], {'op': 'Const', 'type': 'Const', 'value': [0]}),
    **regular_op_with_empty_data('ss_end_gather_2', {'op': 'Gather', 'type': 'Gather'}),
    **valued_const_with_data('ss_end_gather_2_idx', int64_array([2])),
    **regular_op_with_shaped_data('ss_end_gather_2_axis', [], {'op': 'Const', 'type': 'Const', 'value': [0]}),
    **regular_op_with_empty_data('ss_end_gather_3', {'op': 'Gather', 'type': 'Gather'}),
    **valued_const_with_data('ss_end_gather_3_idx', int64_array([3])),
    **regular_op_with_shaped_data('ss_end_gather_3_axis', [], {'op': 'Const', 'type': 'Const', 'value': [0]}),
    **regular_op_with_empty_data('ss_end_const_0', {'op': 'Const', 'type': 'Const', 'value': int64_array([0])}),
    **regular_op_with_empty_data('ss_end_const_1', {'op': 'Const', 'type': 'Const', 'value': int64_array([0])}),
    **regular_op_with_empty_data('ss_end_const_2', {'op': 'Const', 'type': 'Const', 'value': int64_array([0])}),
    **regular_op_with_empty_data('ss_end_const_3', {'op': 'Const', 'type': 'Const', 'value': int64_array([0])}),
    **regular_op_with_empty_data('ss_end_concat', {'op': 'Concat', 'type': 'Concat'}),

    **regular_op_with_empty_data('ss_strides', {'op': 'Const', 'type': 'Const'}),
    **regular_op_with_empty_data('ss', {'op': 'StridedSlice', 'type': 'StridedSlice',
                                        'new_axis_mask': np.zeros(4, dtype=np.int64),
                                        'shrink_axis_mask': np.zeros(4, dtype=np.int64),
                                        'ellipsis_mask': np.zeros(4, dtype=np.int64)}),
    **result('result')
}

pattern_graph = [
    *connect('input:0', '0:slice'),
    *connect('starts:0', '1:slice'),
    *connect('ends:0', '2:slice'),
    *connect('axes:0', '3:slice'),
    *connect('steps:0', '4:slice'),
    *connect('slice:0', '0:result')
]

pattern_ref_graph = [
    *connect('input:0', '0:ss'),
    *connect('starts:0', '0:ss_begin_clamp'),
    *connect('ss_begin_clamp:0', '0:ss_begin_cast'),
    *connect('ss_begin_clamp_min:0', '1:ss_begin_clamp'),
    *connect('ss_begin_clamp_max:0', '2:ss_begin_clamp'),
    *connect('ss_begin_concat:0', '1:ss'),
    *connect('ends:0', '0:ss_end_clamp'),
    *connect('ss_end_clamp:0', '0:ss_end_cast'),
    *connect('ss_end_clamp_min:0', '1:ss_end_clamp'),
    *connect('ss_end_clamp_max:0', '2:ss_end_clamp'),
    *connect('ss_end_concat:0', '2:ss'),
    *connect('ss_strides:0', '3:ss'),
    *connect('ss:0', '0:result'),

    *connect('ss_begin_gather_0_idx:0', '1:ss_begin_gather_0'),
    *connect('ss_begin_gather_0_axis:0', '2:ss_begin_gather_0'),
    *connect('ss_begin_gather_1_idx:0', '1:ss_begin_gather_1'),
    *connect('ss_begin_gather_1_axis:0', '2:ss_begin_gather_1'),
    *connect('ss_begin_gather_2_idx:0', '1:ss_begin_gather_2'),
    *connect('ss_begin_gather_2_axis:0', '2:ss_begin_gather_2'),
    *connect('ss_begin_gather_3_idx:0', '1:ss_begin_gather_3'),
    *connect('ss_begin_gather_3_axis:0', '2:ss_begin_gather_3'),

    *connect('ss_end_gather_0_idx:0', '1:ss_end_gather_0'),
    *connect('ss_end_gather_0_axis:0', '2:ss_end_gather_0'),
    *connect('ss_end_gather_1_idx:0', '1:ss_end_gather_1'),
    *connect('ss_end_gather_1_axis:0', '2:ss_end_gather_1'),
    *connect('ss_end_gather_2_idx:0', '1:ss_end_gather_2'),
    *connect('ss_end_gather_2_axis:0', '2:ss_end_gather_2'),
    *connect('ss_end_gather_3_idx:0', '1:ss_end_gather_3'),
    *connect('ss_end_gather_3_axis:0', '2:ss_end_gather_3'),
]


class ConvertSliceTests(unittest.TestCase):

    def test_convert_slice_to_strided_slice_one_axis(self):
        graph = build_graph(
            nodes_attrs=nodes_attributes,
            edges=pattern_graph,
            update_attributes={
                'starts': {'value': int64_array([0]), 'shape': [1]},
                'ends': {'value': int64_array([1]), 'shape': [1]},
                'axes': {'value': int64_array([0]), 'shape': [1]},
                'axes_d': {'value': int64_array([0]), 'shape': [1]},
                'steps': {'value': int64_array([1]), 'shape': [1]},
                'steps_d': {'value': int64_array([1]), 'shape': [1]}
            },
            nodes_with_edges_only=True
        )

        ref_graph = build_graph(
            nodes_attrs=nodes_attributes,
            edges=pattern_ref_graph + [
                *connect('ss_begin_cast:0', '0:ss_begin_gather_0'),
                *connect('ss_begin_gather_0:0', '0:ss_begin_concat'),
                *connect('ss_begin_const_1:0', '1:ss_begin_concat'),
                *connect('ss_begin_const_2:0', '2:ss_begin_concat'),
                *connect('ss_begin_const_3:0', '3:ss_begin_concat'),

                *connect('ss_end_cast:0', '0:ss_end_gather_0'),
                *connect('ss_end_gather_0:0', '0:ss_end_concat'),
                *connect('ss_end_const_1:0', '1:ss_end_concat'),
                *connect('ss_end_const_2:0', '2:ss_end_concat'),
                *connect('ss_end_const_3:0', '3:ss_end_concat'),
            ],
            update_attributes={
                'starts': {'value': int64_array([0]), 'shape': [1]},
                'ends': {'value': int64_array([1]), 'shape': [1]},
                'ss_strides': {'value': int64_array([1, 1, 1, 1]), 'shape': [4]},
                'ss': {'begin_mask': int64_array([1, 0, 0, 0]), 'end_mask': int64_array([1, 0, 0, 0])}
            }
        )
        ConvertSlice().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_convert_slice_to_strided_slice_one_axis_steps_is_2(self):
        graph = build_graph(
            nodes_attrs=nodes_attributes,
            edges=pattern_graph,
            update_attributes={
                'starts': {'value': int64_array([0]), 'shape': [1]},
                'ends': {'value': int64_array([150]), 'shape': [1]},
                'axes': {'value': int64_array([2]), 'shape': [1]},
                'axes_d': {'value': int64_array([2]), 'shape': [1]},
                'steps': {'value': int64_array([2]), 'shape': [1]},
                'steps_d': {'value': int64_array([2]), 'shape': [1]}
            },
            nodes_with_edges_only=True
        )

        ref_graph = build_graph(
            nodes_attrs=nodes_attributes,
            edges=pattern_ref_graph + [
                *connect('ss_begin_cast:0', '0:ss_begin_gather_0'),
                *connect('ss_begin_gather_0:0', '2:ss_begin_concat'),
                *connect('ss_begin_const_0:0', '0:ss_begin_concat'),
                *connect('ss_begin_const_1:0', '1:ss_begin_concat'),
                *connect('ss_begin_const_3:0', '3:ss_begin_concat'),

                *connect('ss_end_cast:0', '0:ss_end_gather_0'),
                *connect('ss_end_gather_0:0', '2:ss_end_concat'),
                *connect('ss_end_const_0:0', '0:ss_end_concat'),
                *connect('ss_end_const_1:0', '1:ss_end_concat'),
                *connect('ss_end_const_3:0', '3:ss_end_concat'),
            ],
            update_attributes={
                'starts': {'value': int64_array([0]), 'shape': [1]},
                'ends': {'value': int64_array([150]), 'shape': [1]},
                'ss_strides': {'value': int64_array([1, 1, 2, 1]), 'shape': [4]},
                'ss': {'begin_mask': int64_array([0, 0, 1, 0]), 'end_mask': int64_array([0, 0, 1, 0])}
            }
        )
        ConvertSlice().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_convert_slice_to_strided_slice_two_axes(self):
        graph = build_graph(
            nodes_attrs=nodes_attributes,
            edges=pattern_graph,
            update_attributes={
                'starts': {'value': int64_array([0, 0]), 'shape': [2]},
                'ends': {'value': int64_array([150, 150]), 'shape': [2]},
                'axes': {'value': int64_array([2, 3]), 'shape': [2]},
                'axes_d': {'value': int64_array([2, 3]), 'shape': [2]},
                'steps': {'value': int64_array([1, 1]), 'shape': [2]},
                'steps_d': {'value': int64_array([1, 1]), 'shape': [2]}
            },
            nodes_with_edges_only=True
        )

        ref_graph = build_graph(
            nodes_attrs=nodes_attributes,
            edges=pattern_ref_graph + [
                *connect('ss_begin_cast:0', '0:ss_begin_gather_0'),
                *connect('ss_begin_gather_0:0', '2:ss_begin_concat'),
                *connect_data('ss_begin_cast:0', '0:ss_begin_gather_1'),
                *connect('ss_begin_gather_1:0', '3:ss_begin_concat'),
                *connect('ss_begin_const_0:0', '0:ss_begin_concat'),
                *connect('ss_begin_const_1:0', '1:ss_begin_concat'),

                *connect('ss_end_cast:0', '0:ss_end_gather_0'),
                *connect('ss_end_gather_0:0', '2:ss_end_concat'),
                *connect_data('ss_end_cast:0', '0:ss_end_gather_1'),
                *connect('ss_end_gather_1:0', '3:ss_end_concat'),
                *connect('ss_end_const_0:0', '0:ss_end_concat'),
                *connect('ss_end_const_1:0', '1:ss_end_concat'),
            ],
            update_attributes={
                'starts': {'value': int64_array([0, 0]), 'shape': [2]},
                'ends': {'value': int64_array([150, 150]), 'shape': [2]},
                'ss_strides': {'value': int64_array([1, 1, 1, 1]), 'shape': [4]},
                'ss': {'begin_mask': int64_array([0, 0, 1, 1]), 'end_mask': int64_array([0, 0, 1, 1])}
            }
        )
        ConvertSlice().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_convert_slice_to_strided_slice_three_axes(self):
        graph = build_graph(
            nodes_attrs=nodes_attributes,
            edges=pattern_graph,
            update_attributes={
                'starts': {'value': int64_array([0, 0, 0]), 'shape': [3]},
                'ends': {'value': int64_array([2, 150, 150]), 'shape': [3]},
                'axes': {'value': int64_array([1, 2, 3]), 'shape': [3]},
                'axes_d': {'value': int64_array([1, 2, 3]), 'shape': [3]},
                'steps': {'value': int64_array([1, 1, 1]), 'shape': [3]},
                'steps_d': {'value': int64_array([1, 1, 1]), 'shape': [3]}
            },
            nodes_with_edges_only=True
        )

        ref_graph = build_graph(
            nodes_attrs=nodes_attributes,
            edges=pattern_ref_graph + [
                *connect('ss_begin_cast:0', '0:ss_begin_gather_0'),
                *connect('ss_begin_gather_0:0', '1:ss_begin_concat'),
                *connect_data('ss_begin_cast:0', '0:ss_begin_gather_1'),
                *connect('ss_begin_gather_1:0', '2:ss_begin_concat'),
                *connect_data('ss_begin_cast:0', '0:ss_begin_gather_2'),
                *connect('ss_begin_gather_2:0', '3:ss_begin_concat'),
                *connect('ss_begin_const_0:0', '0:ss_begin_concat'),

                *connect('ss_end_cast:0', '0:ss_end_gather_0'),
                *connect('ss_end_gather_0:0', '1:ss_end_concat'),
                *connect_data('ss_end_cast:0', '0:ss_end_gather_1'),
                *connect('ss_end_gather_1:0', '2:ss_end_concat'),
                *connect_data('ss_end_cast:0', '0:ss_end_gather_2'),
                *connect('ss_end_gather_2:0', '3:ss_end_concat'),
                *connect('ss_end_const_0:0', '0:ss_end_concat'),
            ],
            update_attributes={
                'starts': {'value': int64_array([0, 0, 0]), 'shape': [3]},
                'ends': {'value': int64_array([2, 150, 150]), 'shape': [3]},
                'ss_strides': {'value': int64_array([1, 1, 1, 1]), 'shape': [4]},
                'ss': {'begin_mask': int64_array([0, 1, 1, 1]), 'end_mask': int64_array([0, 1, 1, 1])}
            }
        )
        ConvertSlice().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_convert_slice_to_strided_slice_not_sorted_axes(self):
        graph = build_graph(
            nodes_attrs=nodes_attributes,
            edges=pattern_graph,
            update_attributes={
                'starts': {'value': int64_array([0, 1, 1, 0]), 'shape': [4]},
                'ends': {'value': int64_array([1, 150, 150, 2]), 'shape': [4]},
                'axes': {'value': int64_array([0, 2, 3, 1]), 'shape': [4]},
                'axes_d': {'value': int64_array([0, 2, 3, 1]), 'shape': [4]},
                'steps': {'value': int64_array([1, 1, 1, 1]), 'shape': [4]},
                'steps_d': {'value': int64_array([1, 1, 1, 1]), 'shape': [4]}
            },
            nodes_with_edges_only=True
        )

        ref_graph = build_graph(
            nodes_attrs=nodes_attributes,
            edges=pattern_ref_graph + [
                *connect('ss_begin_cast:0', '0:ss_begin_gather_0'),
                *connect('ss_begin_gather_0:0', '0:ss_begin_concat'),
                *connect_data('ss_begin_cast:0', '0:ss_begin_gather_1'),
                *connect('ss_begin_gather_1:0', '2:ss_begin_concat'),
                *connect_data('ss_begin_cast:0', '0:ss_begin_gather_2'),
                *connect('ss_begin_gather_2:0', '3:ss_begin_concat'),
                *connect_data('ss_begin_cast:0', '0:ss_begin_gather_3'),
                *connect('ss_begin_gather_3:0', '1:ss_begin_concat'),

                *connect('ss_end_cast:0', '0:ss_end_gather_0'),
                *connect('ss_end_gather_0:0', '0:ss_end_concat'),
                *connect_data('ss_end_cast:0', '0:ss_end_gather_1'),
                *connect('ss_end_gather_1:0', '2:ss_end_concat'),
                *connect_data('ss_end_cast:0', '0:ss_end_gather_2'),
                *connect('ss_end_gather_2:0', '3:ss_end_concat'),
                *connect_data('ss_end_cast:0', '0:ss_end_gather_3'),
                *connect('ss_end_gather_3:0', '1:ss_end_concat'),
            ],
            update_attributes={
                'starts': {'value': int64_array([0, 1, 1, 0]), 'shape': [4]},
                'ends': {'value': int64_array([1, 150, 150, 2]), 'shape': [4]},
                'ss_strides': {'value': int64_array([1, 1, 1, 1]), 'shape': [4]},
                'ss': {'begin_mask': int64_array([1, 1, 1, 1]), 'end_mask': int64_array([1, 1, 1, 1])}
            }
        )
        ConvertSlice().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_convert_slice_to_strided_slice_without_axes_and_steps(self):
        graph = build_graph(
            nodes_attrs=nodes_attributes,
            edges=[
                *connect('input:0', '0:slice'),
                *connect('starts:0', '1:slice'),
                *connect('ends:0', '2:slice'),
                *connect('slice:0', '0:result')
            ],
            update_attributes={
                'starts': {'value': int64_array([0, 0, 0, 0]), 'shape': [4]},
                'ends': {'value': int64_array([1, 2, 150, 150]), 'shape': [4]},
            },
            nodes_with_edges_only=True
        )

        ref_graph = build_graph(
            nodes_attrs=nodes_attributes,
            edges=pattern_ref_graph + [
                *connect('ss_begin_cast:0', '0:ss_begin_gather_0'),
                *connect('ss_begin_gather_0:0', '0:ss_begin_concat'),
                *connect_data('ss_begin_cast:0', '0:ss_begin_gather_1'),
                *connect('ss_begin_gather_1:0', '1:ss_begin_concat'),
                *connect_data('ss_begin_cast:0', '0:ss_begin_gather_2'),
                *connect('ss_begin_gather_2:0', '2:ss_begin_concat'),
                *connect_data('ss_begin_cast:0', '0:ss_begin_gather_3'),
                *connect('ss_begin_gather_3:0', '3:ss_begin_concat'),

                *connect('ss_end_cast:0', '0:ss_end_gather_0'),
                *connect('ss_end_gather_0:0', '0:ss_end_concat'),
                *connect_data('ss_end_cast:0', '0:ss_end_gather_1'),
                *connect('ss_end_gather_1:0', '1:ss_end_concat'),
                *connect_data('ss_end_cast:0', '0:ss_end_gather_2'),
                *connect('ss_end_gather_2:0', '2:ss_end_concat'),
                *connect_data('ss_end_cast:0', '0:ss_end_gather_3'),
                *connect('ss_end_gather_3:0', '3:ss_end_concat'),
            ],
            update_attributes={
                'starts': {'value': int64_array([0, 0, 0, 0]), 'shape': [4]},
                'ends': {'value': int64_array([1, 2, 150, 150]), 'shape': [4]},
                'ss_strides': {'value': int64_array([1, 1, 1, 1]), 'shape': [4]},
                'ss': {'begin_mask': int64_array([1, 1, 1, 1]), 'end_mask': int64_array([1, 1, 1, 1])}
            }
        )
        ConvertSlice().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)
