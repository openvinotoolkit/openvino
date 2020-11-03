"""
 Copyright (C) 2018-2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import unittest

import numpy as np
from generator import generate, generator

from extensions.middle.SliceConverter import ConvertSlice
from mo.front.common.partial_infer.utils import int64_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph, regular_op_with_shaped_data, valued_const_with_data, \
    regular_op_with_empty_data, result, connect, const, empty_data


@generator
class ConvertSliceTests(unittest.TestCase):
    @generate(*[
        (int64_array([1, 3, 300, 300]), np.array([0, 0]), np.array([150, 150]), np.array([2, 3]), np.array([1, 1]),
         (int64_array([0, 0]), int64_array([])), (int64_array([0, 0]), int64_array([])), int64_array([1, 1, 1, 1]),
         int64_array([0, 0, 1, 1]), int64_array([0, 0, 1, 1])),

        (int64_array([1, 3, 300, 300]), np.array([0]), np.array([150]), np.array([2]), np.array([1]),
         (int64_array([0, 0]), int64_array([0])), (int64_array([0, 0]), int64_array([0])), int64_array([1, 1, 1, 1]),
         int64_array([0, 0, 1, 0]), int64_array([0, 0, 1, 0])),

        (int64_array([1, 3, 300, 300]), np.array([0, 0]), np.array([150, 150]), np.array([-2, -1]), np.array([1, 1]),
         (int64_array([0, 0]), int64_array([])), (int64_array([0, 0]), int64_array([])), int64_array([1, 1, 1, 1]),
         int64_array([0, 0, 1, 1]), int64_array([0, 0, 1, 1]))
    ])
    def test_convert_slice_to_strided_slice(self, input_shape, start, end, axes, steps,
                                            ss_begin_parts: tuple, ss_end_parts: tuple, ss_steps,
                                            ss_begin_mask, ss_end_mask):
        graph = build_graph(
            nodes_attrs={
                **regular_op_with_shaped_data('input', input_shape, {'type': 'Parameter'}),
                **valued_const_with_data('start', start),
                **valued_const_with_data('end', end),
                **valued_const_with_data('axes', axes),
                **valued_const_with_data('steps', steps),
                **regular_op_with_empty_data('slice', {'type': None, 'op': 'Slice'}),
                **result('result')
            },
            edges=[
                *connect('input', 'slice'),
                *connect('start', '1:slice'),
                *connect('end', '2:slice'),
                *connect('axes', '3:slice'),
                *connect('steps', '4:slice'),
                *connect('slice', 'result')
            ]
        )
        ref_graph = build_graph(
            nodes_attrs={
                **regular_op_with_shaped_data('input', input_shape, {'type': 'Parameter'}),
                **valued_const_with_data('start', start),
                **valued_const_with_data('begin_first_part', ss_begin_parts[0]),
                **valued_const_with_data('begin_last_part', ss_begin_parts[1]),
                **regular_op_with_empty_data('convert_start', {'op': 'Cast', 'type': 'Convert', 'dst_type': np.int64}),
                **regular_op_with_empty_data('ss_begin', {'type': 'Concat', 'op': 'Concat', 'axis': 0}),
                **valued_const_with_data('end', end),
                **valued_const_with_data('end_first_part', ss_end_parts[0]),
                **valued_const_with_data('end_last_part', ss_end_parts[1]),
                **regular_op_with_empty_data('convert_end', {'op': 'Cast', 'type': 'Convert', 'dst_type': np.int64}),
                **regular_op_with_empty_data('ss_end', {'type': 'Concat', 'op': 'Concat', 'axis': 0}),
                **const('ss_steps', ss_steps),
                **empty_data('ss_steps_d'),
                **regular_op_with_empty_data('ss', {'op': 'StridedSlice', 'type': 'StridedSlice',
                                                    'begin_mask': ss_begin_mask, 'end_mask': ss_end_mask,
                                                    'new_axis_mask': np.zeros(len(input_shape), dtype=np.int64),
                                                    'shrink_axis_mask': np.zeros(len(input_shape), dtype=np.int64),
                                                    'ellipsis_mask': np.zeros(len(input_shape), dtype=np.int64)}),
                **result('result')
            },
            edges=[
                *connect('input', 'ss'),
                *connect('begin_first_part', 'ss_begin'),
                *connect('start', 'convert_start'),
                *connect('convert_start', '1:ss_begin'),
                *connect('begin_last_part', '2:ss_begin'),
                *connect('ss_begin', '1:ss'),
                *connect('end_first_part', 'ss_end'),
                *connect('end', 'convert_end'),
                *connect('convert_end', '1:ss_end'),
                *connect('end_last_part', '2:ss_end'),
                *connect('ss_end', '2:ss'),
                *connect('ss_steps', '3:ss'),
                *connect('ss', 'result')
            ]
        )
        ConvertSlice().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_convert_slice_to_strided_slice_without_axes_and_steps(self):
        graph = build_graph(
            nodes_attrs={
                **regular_op_with_shaped_data('input', int64_array([2, 5, 10]), {'type': 'Parameter'}),
                **valued_const_with_data('start', np.array([0, 0, 0])),
                **valued_const_with_data('end', np.array([1, 3, 5])),
                **regular_op_with_empty_data('slice', {'type': None, 'op': 'Slice'}),
                **result('result')
            },
            edges=[
                *connect('input', 'slice'),
                *connect('start', '1:slice'),
                *connect('end', '2:slice'),
                *connect('slice', 'result')
            ]
        )
        ref_graph = build_graph(
            nodes_attrs={
                **regular_op_with_shaped_data('input', int64_array([2, 5, 10]), {'type': 'Parameter'}),
                **valued_const_with_data('start', np.array([0, 0, 0])),
                **valued_const_with_data('begin_first_part', int64_array([])),
                **valued_const_with_data('begin_last_part', int64_array([])),
                **regular_op_with_empty_data('convert_start', {'op': 'Cast', 'type': 'Convert', 'dst_type': np.int64}),
                **regular_op_with_empty_data('ss_begin', {'type': 'Concat', 'op': 'Concat', 'axis': 0}),
                **valued_const_with_data('end', np.array([1, 3, 5])),
                **valued_const_with_data('end_first_part', int64_array([])),
                **valued_const_with_data('end_last_part', int64_array([])),
                **regular_op_with_empty_data('convert_end', {'op': 'Cast', 'type': 'Convert', 'dst_type': np.int64}),
                **regular_op_with_empty_data('ss_end', {'type': 'Concat', 'op': 'Concat', 'axis': 0}),
                **const('ss_steps', int64_array([1, 1, 1])),
                **empty_data('ss_steps_d'),
                **regular_op_with_empty_data('ss', {'op': 'StridedSlice', 'type': 'StridedSlice',
                                                    'begin_mask': int64_array([1, 1, 1]), 'end_mask': int64_array([1, 1, 1]),
                                                    'new_axis_mask': np.zeros(3, dtype=np.int64),
                                                    'shrink_axis_mask': np.zeros(3, dtype=np.int64),
                                                    'ellipsis_mask': np.zeros(3, dtype=np.int64)}),
                **result('result')
            },
            edges=[
                *connect('input', 'ss'),
                *connect('begin_first_part', 'ss_begin'),
                *connect('start', 'convert_start'),
                *connect('convert_start', '1:ss_begin'),
                *connect('begin_last_part', '2:ss_begin'),
                *connect('ss_begin', '1:ss'),
                *connect('end_first_part', 'ss_end'),
                *connect('end', 'convert_end'),
                *connect('convert_end', '1:ss_end'),
                *connect('end_last_part', '2:ss_end'),
                *connect('ss_end', '2:ss'),
                *connect('ss_steps', '3:ss'),
                *connect('ss', 'result')
            ]
        )
        ConvertSlice().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)
