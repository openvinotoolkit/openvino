"""
 Copyright (c) 2020 Intel Corporation

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

from extensions.middle.SplitConcatPairToInterpolate import SplitConcatPairToInterpolate
from mo.front.common.partial_infer.utils import int64_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph

graph_node_attrs_for_2d_spatial_case = {
        'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
        'placeholder_data': {
            'value': None,
            'shape': int64_array([1, 100, 120, 150]),
            'kind': 'data',
            'data_type': None
        },
        'split': {'type': 'Split', 'kind': 'op', 'op': 'Split', 'num_splits': 3},
        'split_axis_const': {
            'kind': 'op',
            'value': np.array(3, dtype=np.int64),
            'op': 'Const',
            'type': 'Const'
        },
        'split_axis_const_data': {'value': None, 'shape': np.array(3, dtype=np.int64).shape, 'kind': 'data'},
        'concat': {'type': 'Concat', 'kind': 'op', 'axis': 3},
        'split_data_0': {'value': None, 'shape': int64_array([1, 100, 120, 50]), 'kind': 'data'},
        'split_data_1': {'value': None, 'shape': int64_array([1, 100, 120, 50]), 'kind': 'data'},
        'split_data_2': {'value': None, 'shape': int64_array([1, 100, 120, 50]), 'kind': 'data'},
        'concat_data': {'value': None, 'shape': int64_array([1, 100, 120, 300]), 'kind': 'data'},
        'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
        'abs_data': {'value': None, 'shape': int64_array([1, 100, 120, 300]), 'kind': 'data'},
        'output': {'kind': 'op', 'op': 'Result'},
    }


graph_node_attrs_for_3d_spatial_case = {
        'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
        'placeholder_data': {
            'value': None,
            'shape': int64_array([1, 3, 100, 120, 150]),
            'kind': 'data',
            'data_type': None
        },
        'split': {'type': 'Split', 'kind': 'op', 'op': 'Split', 'num_splits': 3},
        'split_axis_const': {
            'kind': 'op',
            'value': np.array(4, dtype=np.int64),
            'op': 'Const',
            'type': 'Const'
        },
        'split_axis_const_data': {'value': None, 'shape': np.array(4, dtype=np.int64).shape, 'kind': 'data'},
        'concat': {'type': 'Concat', 'kind': 'op', 'axis': 4},
        'split_data_0': {'value': None, 'shape': int64_array([1, 3, 100, 120, 50]), 'kind': 'data'},
        'split_data_1': {'value': None, 'shape': int64_array([1, 3, 100, 120, 50]), 'kind': 'data'},
        'split_data_2': {'value': None, 'shape': int64_array([1, 3, 100, 120, 50]), 'kind': 'data'},
        'concat_data': {'value': None, 'shape': int64_array([1, 3, 100, 120, 300]), 'kind': 'data'},
        'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
        'abs_data': {'value': None, 'shape': int64_array([1, 3, 100, 120, 300]), 'kind': 'data'},
        'output': {'kind': 'op', 'op': 'Result'},
    }


graph_edges = [
        ('placeholder', 'placeholder_data'),
        ('placeholder_data', 'split', {'in': 0}),
        ('split_axis_const', 'split_axis_const_data'),
        ('split_axis_const_data', 'split', {'in': 1}),
        ('split', 'split_data_0', {'out': 0}),
        ('split', 'split_data_1', {'out': 1}),
        ('split', 'split_data_2', {'out': 2}),
        ('split_data_0', 'concat', {'in': 0}),
        ('split_data_0', 'concat', {'in': 1}),
        ('split_data_1', 'concat', {'in': 2}),
        ('split_data_1', 'concat', {'in': 3}),
        ('split_data_2', 'concat', {'in': 4}),
        ('split_data_2', 'concat', {'in': 5}),
        ('concat', 'concat_data'),
        ('concat_data', 'abs'),
        ('abs', 'abs_data'),
        ('abs_data', 'output')
    ]


ref_graph_edges = [
        ('placeholder', 'placeholder_data'),
        ('placeholder_data', 'interpolate', {'in': 0}),
        ('placeholder_data', 'shape'),
        ('shape', 'sslice', {'in': 0}),
        ('slice_begin', 'sslice', {'in': 1}),
        ('slice_end', 'sslice', {'in': 2}),
        ('sslice', 'sslice_data'),
        ('scales', 'scales_data'),
        ('sslice_data', 'mul', {'in': 0}),
        ('scales_data', 'mul', {'in': 1}),
        ('mul', 'mul_data'),
        ('mul_data', 'interpolate', {'in': 1}),
        ('interpolate', 'interpolate_data'),
        ('interpolate_data', 'abs'),
        ('abs', 'abs_data'),
        ('abs_data', 'output'),
    ]


ref_graph_node_attrs_for_2d_spatial_case_1 = {
        'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
        'placeholder_data': {
            'value': None,
            'shape': int64_array([1, 100, 120, 150]),
            'kind': 'data',
            'data_type': None
        },
        'interpolate': {
            'type': 'Interpolate',
            'kind': 'op',
            'op': 'Interpolate',
            'axes': int64_array([3]),
            'mode': 'nearest'
        },
        'shape': {'type': 'ShapeOf', 'kind': 'op', 'op': 'ShapeOf'},
        'slice_begin': {
            'type': 'Const',
            'op': 'Const',
            'kind': 'op',
            'value': int64_array([3]),
            'shape': int64_array([1])
        },
        'slice_end': {'type': 'Const', 'op': 'Const', 'kind': 'op', 'value': int64_array([4])},
        'sslice': {
            'kind': 'op',
            'type': 'StridedSlice',
            'op': 'StridedSlice',
            'begin_mask': int64_array([1]),
            'end_mask': int64_array([1]),
            'new_axis_mask': int64_array([0]),
            'shrink_axis_mask': int64_array([0]),
            'ellipsis_mask': int64_array([0]),
        },
        'sslice_data': {'kind': 'data', 'shape': None},
        'scales': {
            'type': 'Const',
            'op': 'Const',
            'kind': 'op',
            'value': int64_array([2]),
            'shape': int64_array([1])
        },
        'scales_data': {'kind': 'data', 'shape': None},
        'mul': {'kind': 'op', 'op': 'Mul', 'type': 'Multiply'},
        'mul_data': {'kind': 'data', 'shape': None},
        'interpolate_data': {'value': None, 'shape': int64_array([1, 100, 120, 300]), 'kind': 'data'},
        'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
        'abs_data': {'value': None, 'shape': int64_array([1, 100, 120, 300]), 'kind': 'data'},
        'output': {'kind': 'op', 'op': 'Result'},
    }

ref_graph_node_attrs_for_2d_spatial_case_2 = {
        'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
        'placeholder_data': {
            'value': None,
            'shape': int64_array([1, 100, 120, 150]),
            'kind': 'data',
            'data_type': None
        },
        'interpolate': {
            'type': 'Interpolate',
            'kind': 'op',
            'op': 'Interpolate',
            'axes': int64_array([2]),
            'mode': 'nearest'
        },
        'shape': {'type': 'ShapeOf', 'kind': 'op', 'op': 'ShapeOf'},
        'slice_begin': {
            'type': 'Const',
            'op': 'Const',
            'kind': 'op',
            'value': int64_array([2]),
            'shape': int64_array([1])
        },
        'slice_end': {'type': 'Const', 'op': 'Const', 'kind': 'op', 'value': int64_array([3])},
        'sslice': {
            'kind': 'op',
            'type': 'StridedSlice',
            'op': 'StridedSlice',
            'begin_mask': int64_array([1]),
            'end_mask': int64_array([1]),
            'new_axis_mask': int64_array([0]),
            'shrink_axis_mask': int64_array([0]),
            'ellipsis_mask': int64_array([0]),
        },
        'sslice_data': {'kind': 'data', 'shape': None},
        'scales': {
            'type': 'Const',
            'op': 'Const',
            'kind': 'op',
            'value': int64_array([2]),
            'shape': int64_array([1])
        },
        'scales_data': {'kind': 'data', 'shape': None},
        'mul': {'kind': 'op', 'op': 'Mul', 'type': 'Multiply'},
        'mul_data': {'kind': 'data', 'shape': None},
        'interpolate_data': {'value': None, 'shape': int64_array([1, 100, 240, 150]), 'kind': 'data'},
        'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
        'abs_data': {'value': None, 'shape': int64_array([1, 100, 240, 150]), 'kind': 'data'},
        'output': {'kind': 'op', 'op': 'Result'},
    }


ref_graph_node_attrs_for_3d_spatial_case_1 = {
        'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
        'placeholder_data': {
            'value': None,
            'shape': int64_array([1, 3, 100, 120, 150]),
            'kind': 'data',
            'data_type': None
        },
        'interpolate': {
            'type': 'Interpolate',
            'kind': 'op',
            'op': 'Interpolate',
            'axes': int64_array([4]),
            'mode': 'nearest'
        },
        'shape': {'type': 'ShapeOf', 'kind': 'op', 'op': 'ShapeOf'},
        'slice_begin': {
            'type': 'Const',
            'op': 'Const',
            'kind': 'op',
            'value': int64_array([4]),
            'shape': int64_array([1])
        },
        'slice_end': {'type': 'Const', 'op': 'Const', 'kind': 'op', 'value': int64_array([5])},
        'sslice': {
            'kind': 'op',
            'type': 'StridedSlice',
            'op': 'StridedSlice',
            'begin_mask': int64_array([1]),
            'end_mask': int64_array([1]),
            'new_axis_mask': int64_array([0]),
            'shrink_axis_mask': int64_array([0]),
            'ellipsis_mask': int64_array([0]),
        },
        'sslice_data': {'kind': 'data', 'shape': None},
        'scales': {
            'type': 'Const',
            'op': 'Const',
            'kind': 'op',
            'value': int64_array([2]),
            'shape': int64_array([1])
        },
        'scales_data': {'kind': 'data', 'shape': None},
        'mul': {'kind': 'op', 'op': 'Mul', 'type': 'Multiply'},
        'mul_data': {'kind': 'data', 'shape': None},
        'interpolate_data': {'value': None, 'shape': int64_array([1, 3, 100, 120, 300]), 'kind': 'data'},
        'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
        'abs_data': {'value': None, 'shape': int64_array([1, 3, 100, 120, 300]), 'kind': 'data'},
        'output': {'kind': 'op', 'op': 'Result'},
    }


ref_graph_node_attrs_for_3d_spatial_case_2 = {
        'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
        'placeholder_data': {
            'value': None,
            'shape': int64_array([1, 3, 100, 120, 150]),
            'kind': 'data',
            'data_type': None
        },
        'interpolate': {
            'type': 'Interpolate',
            'kind': 'op',
            'op': 'Interpolate',
            'axes': int64_array([3]),
            'mode': 'nearest'
        },
        'shape': {'type': 'ShapeOf', 'kind': 'op', 'op': 'ShapeOf'},
        'slice_begin': {
            'type': 'Const',
            'op': 'Const',
            'kind': 'op',
            'value': int64_array([4]),
            'shape': int64_array([1])
        },
        'slice_end': {'type': 'Const', 'op': 'Const', 'kind': 'op', 'value': int64_array([5])},
        'sslice': {
            'kind': 'op',
            'type': 'StridedSlice',
            'op': 'StridedSlice',
            'begin_mask': int64_array([1]),
            'end_mask': int64_array([1]),
            'new_axis_mask': int64_array([0]),
            'shrink_axis_mask': int64_array([0]),
            'ellipsis_mask': int64_array([0]),
        },
        'sslice_data': {'kind': 'data', 'shape': None},
        'scales': {
            'type': 'Const',
            'op': 'Const',
            'kind': 'op',
            'value': int64_array([2]),
            'shape': int64_array([1])
        },
        'scales_data': {'kind': 'data', 'shape': None},
        'mul': {'kind': 'op', 'op': 'Mul', 'type': 'Multiply'},
        'mul_data': {'kind': 'data', 'shape': None},
        'interpolate_data': {'value': None, 'shape': int64_array([1, 3, 100, 240, 150]), 'kind': 'data'},
        'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
        'abs_data': {'value': None, 'shape': int64_array([1, 3, 100, 240, 150]), 'kind': 'data'},
        'output': {'kind': 'op', 'op': 'Result'},
    }


class SplitConcatPairToInterpolateTest(unittest.TestCase):
    def test_spatial_2d_split_concat_1(self):
        graph = build_graph(
            nodes_attrs=graph_node_attrs_for_2d_spatial_case,
            edges=graph_edges
        )
        ref_graph = build_graph(
            nodes_attrs=ref_graph_node_attrs_for_2d_spatial_case_1,
            edges=ref_graph_edges
        )
        SplitConcatPairToInterpolate().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output')
        self.assertTrue(flag, resp)

    def test_spatial_2d_split_concat_2(self):
        graph = build_graph(
            nodes_attrs=graph_node_attrs_for_2d_spatial_case,
            edges=graph_edges,
            update_attributes={
                'split': {'type': 'Split', 'kind': 'op', 'op': 'Split', 'num_splits': 3},
                'split_axis_const': {
                    'kind': 'op',
                    'value': np.array(2, dtype=np.int64),
                    'op': 'Const',
                    'type': 'Const'
                },
                'split_axis_const_data': {'value': None, 'shape': np.array(2, dtype=np.int64).shape, 'kind': 'data'},
                'concat': {'type': 'Concat', 'kind': 'op', 'axis': 2},
                'split_data_0': {'value': None, 'shape': int64_array([1, 100, 40, 150]), 'kind': 'data'},
                'split_data_1': {'value': None, 'shape': int64_array([1, 100, 40, 150]), 'kind': 'data'},
                'split_data_2': {'value': None, 'shape': int64_array([1, 100, 40, 150]), 'kind': 'data'},
                'concat_data': {'value': None, 'shape': int64_array([1, 100, 240, 150]), 'kind': 'data'},
                'abs_data': {'value': None, 'shape': int64_array([1, 100, 240, 150]), 'kind': 'data'},
            }
        )
        ref_graph = build_graph(
            nodes_attrs=ref_graph_node_attrs_for_2d_spatial_case_2,
            edges=ref_graph_edges
        )
        SplitConcatPairToInterpolate().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output')
        self.assertTrue(flag, resp)

    def test_spatial_3d_split_concat_1(self):
        graph = build_graph(
            nodes_attrs=graph_node_attrs_for_3d_spatial_case,
            edges=graph_edges
        )
        ref_graph = build_graph(
            nodes_attrs=ref_graph_node_attrs_for_3d_spatial_case_1,
            edges=ref_graph_edges
        )
        SplitConcatPairToInterpolate().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output')
        self.assertTrue(flag, resp)

    def test_spatial_3d_split_concat_2(self):
        graph = build_graph(
            nodes_attrs=graph_node_attrs_for_3d_spatial_case,
            edges=graph_edges,
            update_attributes={
                'split': {'type': 'Split', 'kind': 'op', 'op': 'Split', 'num_splits': 3},
                'split_axis_const': {
                    'kind': 'op',
                    'value': np.array(3, dtype=np.int64),
                    'op': 'Const',
                    'type': 'Const'
                },
                'split_axis_const_data': {'value': None, 'shape': np.array(3, dtype=np.int64).shape, 'kind': 'data'},
                'concat': {'type': 'Concat', 'kind': 'op', 'axis': 3},
                'split_data_0': {'value': None, 'shape': int64_array([1, 3, 100, 40, 150]), 'kind': 'data'},
                'split_data_1': {'value': None, 'shape': int64_array([1, 3, 100, 40, 150]), 'kind': 'data'},
                'split_data_2': {'value': None, 'shape': int64_array([1, 3, 100, 40, 150]), 'kind': 'data'},
                'concat_data': {'value': None, 'shape': int64_array([1, 3, 100, 240, 150]), 'kind': 'data'},
                'abs_data': {'value': None, 'shape': int64_array([1, 3, 100, 240, 150]), 'kind': 'data'},
            }
        )
        ref_graph = build_graph(
            nodes_attrs=ref_graph_node_attrs_for_3d_spatial_case_2,
            edges=ref_graph_edges
        )
        SplitConcatPairToInterpolate().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output')
        self.assertTrue(flag, resp)
