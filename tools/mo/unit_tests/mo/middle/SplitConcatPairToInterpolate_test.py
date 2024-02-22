# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.middle.SplitConcatPairToInterpolate import SplitConcatPairToInterpolate
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph

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
    'split_axis_const_data': {
        'value': np.array(3, dtype=np.int64),
        'shape': np.array(3, dtype=np.int64).shape,
        'kind': 'data'
    },
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
        'split_axis_const_data': {
            'value': np.array(4, dtype=np.int64),
            'shape': np.array(4, dtype=np.int64).shape,
            'kind': 'data'
        },
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


ref_graph_edges_opset4 = [
        ('placeholder', 'placeholder_data'),
        ('placeholder_data', 'interpolate', {'in': 0}),
        ('placeholder_data', 'shape'),
        ('shape', 'shape_data'),
        ('shape_data', 'sslice', {'in': 0}),
        ('slice_begin', 'slice_begin_data'),
        ('slice_begin_data', 'sslice', {'in': 1}),
        ('slice_end', 'slice_end_data'),
        ('slice_end_data', 'sslice', {'in': 2}),
        ('sslice', 'sslice_data'),
        ('sslice_data', 'cast_shape_to_float'),
        ('cast_shape_to_float', 'cast_shape_to_float_data'),
        ('scales', 'scales_data'),
        ('axes', 'axes_data'),
        ('cast_shape_to_float_data', 'mul', {'in': 0}),
        ('scales_data', 'mul', {'in': 1, 'out': 0}),
        ('mul', 'mul_data'),
        ('mul_data', 'floor'),
        ('floor', 'floor_data'),
        ('floor_data', 'cast_mul_to_float'),
        ('cast_mul_to_float', 'cast_mul_to_float_data'),
        ('cast_mul_to_float_data', 'interpolate', {'in': 1}),
        ('scales_data', 'interpolate', {'in': 2, 'out': 0}),
        ('axes_data', 'interpolate', {'in': 3}),
        ('interpolate', 'interpolate_data'),
        ('interpolate_data', 'abs'),
        ('abs', 'abs_data'),
        ('abs_data', 'output'),
    ]

ref_graph_node_attrs_for_2d_spatial_case_1_opset4 = {
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
        'mode': 'nearest',
        'antialias': 0,
        'pads_begin': int64_array([0]),
        'pads_end': int64_array([0]),
        'coordinate_transformation_mode': 'half_pixel',
        'nearest_mode': 'round_prefer_floor',
        'cube_coeff': -0.75,
        'version': 'opset4',
        'shape_calculation_mode': 'scales'
    },
    'shape': {'type': 'ShapeOf', 'kind': 'op', 'op': 'ShapeOf'},
    'shape_data': {'kind': 'data', 'shape': None, 'value': None},
    'slice_begin': {
        'type': 'Const',
        'op': 'Const',
        'kind': 'op',
        'value': int64_array([3]),
        'shape': int64_array([1])
    },
    'slice_begin_data': {'kind': 'data', 'shape': int64_array([1]), 'value': int64_array([3])},
    'slice_end': {'type': 'Const', 'op': 'Const', 'kind': 'op', 'value': int64_array([4]), 'shape': int64_array([1])},
    'slice_end_data': {'kind': 'data', 'value': int64_array([4]), 'shape': int64_array([1])},
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
        'value': np.array([2], dtype=np.float32),
        'shape': int64_array([1])
    },
    'scales_data': {'kind': 'data', 'shape': None},
    'cast_shape_to_float': {'kind': 'op', 'op': 'Cast', 'type': 'Convert', 'dst_type': np.float32},
    'cast_shape_to_float_data': {'kind': 'data', 'shape': None},
    'axes': {
        'type': 'Const',
        'op': 'Const',
        'kind': 'op',
        'value': int64_array([3]),
        'shape': int64_array([1])
    },
    'axes_data': {'kind': 'data', 'shape': None},
    'mul': {'kind': 'op', 'op': 'Mul', 'type': 'Multiply'},
    'mul_data': {'kind': 'data', 'shape': None},
    'floor': {'kind': 'op', 'op': 'Floor', 'type': 'Floor'},
    'floor_data': {'kind': 'data', 'shape': None},
    'cast_mul_to_float': {'kind': 'op', 'op': 'Cast', 'type': 'Convert', 'dst_type': np.int64},
    'cast_mul_to_float_data': {'kind': 'data', 'shape': None},
    'interpolate_data': {'value': None, 'shape': int64_array([1, 100, 120, 300]), 'kind': 'data'},
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'abs_data': {'value': None, 'shape': int64_array([1, 100, 120, 300]), 'kind': 'data'},
    'output': {'kind': 'op', 'op': 'Result'},
}

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
    'shape_data': {'kind': 'data', 'shape': None, 'value': None},
    'slice_begin': {
        'type': 'Const',
        'op': 'Const',
        'kind': 'op',
        'value': int64_array([3]),
        'shape': int64_array([1])
    },
    'slice_begin_data': {'kind': 'data', 'shape': None, 'value': None},
    'slice_end': {'type': 'Const', 'op': 'Const', 'kind': 'op', 'value': int64_array([4])},
    'slice_end_data': {'kind': 'data', 'shape': None, 'value': None},
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
    'axes': {
        'type': 'Const',
        'op': 'Const',
        'kind': 'op',
        'value': int64_array([3]),
        'shape': int64_array([1])
    },
    'axes_data': {'kind': 'data', 'shape': None},
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
        'mode': 'nearest',
        'antialias': 0,
        'pads_begin': int64_array([0]),
        'pads_end': int64_array([0]),
        'coordinate_transformation_mode': 'half_pixel',
        'nearest_mode': 'round_prefer_floor',
        'cube_coeff': -0.75,
        'version': 'opset4',
        'shape_calculation_mode': 'scales'
    },
    'shape': {'type': 'ShapeOf', 'kind': 'op', 'op': 'ShapeOf'},
    'shape_data': {'kind': 'data', 'shape': None, 'value': None},
    'slice_begin': {
        'type': 'Const',
        'op': 'Const',
        'kind': 'op',
        'value': int64_array([2]),
        'shape': int64_array([1])
    },
    'slice_begin_data': {'kind': 'data', 'shape': None, 'value': None},
    'slice_end': {'type': 'Const', 'op': 'Const', 'kind': 'op', 'value': int64_array([3])},
    'slice_end_data': {'kind': 'data', 'shape': None, 'value': None},
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
        'value': np.array([2], dtype=np.float32),
        'shape': int64_array([1])
    },
    'scales_data': {'kind': 'data', 'shape': None},
    'cast_shape_to_float': {'kind': 'op', 'op': 'Cast', 'type': 'Convert', 'dst_type': np.float32},
    'cast_shape_to_float_data': {'kind': 'data', 'shape': None},
    'axes': {
        'type': 'Const',
        'op': 'Const',
        'kind': 'op',
        'value': int64_array([3]),
        'shape': int64_array([1])
    },
    'axes_data': {'kind': 'data', 'shape': None},
    'mul': {'kind': 'op', 'op': 'Mul', 'type': 'Multiply'},
    'mul_data': {'kind': 'data', 'shape': None},
    'floor': {'kind': 'op', 'op': 'Floor', 'type': 'Floor'},
    'floor_data': {'kind': 'data', 'shape': None},
    'cast_mul_to_float': {'kind': 'op', 'op': 'Cast', 'type': 'Convert', 'dst_type': np.int64},
    'cast_mul_to_float_data': {'kind': 'data', 'shape': None},
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
        'mode': 'nearest',
        'antialias': 0,
        'pads_begin': int64_array([0]),
        'pads_end': int64_array([0]),
        'coordinate_transformation_mode': 'half_pixel',
        'nearest_mode': 'round_prefer_floor',
        'cube_coeff': -0.75,
        'version': 'opset4',
        'shape_calculation_mode': 'scales'
    },
    'shape': {'type': 'ShapeOf', 'kind': 'op', 'op': 'ShapeOf'},
    'shape_data': {'kind': 'data', 'shape': None, 'value': None},
    'slice_begin': {
        'type': 'Const',
        'op': 'Const',
        'kind': 'op',
        'value': int64_array([4]),
        'shape': int64_array([1])
    },
    'slice_begin_data': {'kind': 'data', 'shape': None, 'value': None},
    'slice_end': {'type': 'Const', 'op': 'Const', 'kind': 'op', 'value': int64_array([5])},
    'slice_end_data': {'kind': 'data', 'shape': None, 'value': None},
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
        'value': np.array([2], dtype=np.float32),
        'shape': int64_array([1])
    },
    'scales_data': {'kind': 'data', 'shape': None},
    'cast_shape_to_float': {'kind': 'op', 'op': 'Cast', 'type': 'Convert', 'dst_type': np.float32},
    'cast_shape_to_float_data': {'kind': 'data', 'shape': None},
    'axes': {
        'type': 'Const',
        'op': 'Const',
        'kind': 'op',
        'value': int64_array([3]),
        'shape': int64_array([1])
    },
    'axes_data': {'kind': 'data', 'shape': None},
    'mul': {'kind': 'op', 'op': 'Mul', 'type': 'Multiply'},
    'mul_data': {'kind': 'data', 'shape': None},
    'floor': {'kind': 'op', 'op': 'Floor', 'type': 'Floor'},
    'floor_data': {'kind': 'data', 'shape': None},
    'cast_mul_to_float': {'kind': 'op', 'op': 'Cast', 'type': 'Convert', 'dst_type': np.int64},
    'cast_mul_to_float_data': {'kind': 'data', 'shape': None},
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
        'mode': 'nearest',
        'antialias': 0,
        'pads_begin': int64_array([0]),
        'pads_end': int64_array([0]),
        'coordinate_transformation_mode': 'half_pixel',
        'nearest_mode': 'round_prefer_floor',
        'cube_coeff': -0.75,
        'version': 'opset4',
        'shape_calculation_mode': 'scales'
    },
    'shape': {'type': 'ShapeOf', 'kind': 'op', 'op': 'ShapeOf'},
    'shape_data': {'kind': 'data', 'shape': None, 'value': None},
    'slice_begin': {
        'type': 'Const',
        'op': 'Const',
        'kind': 'op',
        'value': int64_array([4]),
        'shape': int64_array([1])
    },
    'slice_begin_data': {'kind': 'data', 'shape': None, 'value': None},
    'slice_end': {'type': 'Const', 'op': 'Const', 'kind': 'op', 'value': int64_array([5])},
    'slice_end_data': {'kind': 'data', 'shape': None, 'value': None},
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
        'value': np.array([2], dtype=np.float32),
        'shape': int64_array([1])
    },
    'scales_data': {'kind': 'data', 'shape': None},
    'cast_shape_to_float': {'kind': 'op', 'op': 'Cast', 'type': 'Convert', 'dst_type': np.float32},
    'cast_shape_to_float_data': {'kind': 'data', 'shape': None},
    'axes': {
        'type': 'Const',
        'op': 'Const',
        'kind': 'op',
        'value': int64_array([3]),
        'shape': int64_array([1])
    },
    'axes_data': {'kind': 'data', 'shape': None},
    'mul': {'kind': 'op', 'op': 'Mul', 'type': 'Multiply'},
    'mul_data': {'kind': 'data', 'shape': None},
    'floor': {'kind': 'op', 'op': 'Floor', 'type': 'Floor'},
    'floor_data': {'kind': 'data', 'shape': None},
    'cast_mul_to_float': {'kind': 'op', 'op': 'Cast', 'type': 'Convert', 'dst_type': np.int64},
    'cast_mul_to_float_data': {'kind': 'data', 'shape': None},
    'interpolate_data': {'value': None, 'shape': int64_array([1, 3, 100, 240, 150]), 'kind': 'data'},
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'abs_data': {'value': None, 'shape': int64_array([1, 3, 100, 240, 150]), 'kind': 'data'},
    'output': {'kind': 'op', 'op': 'Result'},
}


graph_node_attrs_when_there_are_two_splits_one_concat = {
    'placeholder1': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder1_data': {
        'value': None,
        'shape': int64_array([1, 13, 13, 3, 2]),
        'kind': 'data',
        'data_type': None
    },
    'placeholder2': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder2_data': {
        'value': None,
        'shape': int64_array([1, 13, 13, 3, 2]),
        'kind': 'data',
        'data_type': None
    },
    'split1': {'type': 'Split', 'kind': 'op', 'op': 'Split', 'num_splits': 2},
    'split1_axis_const': {
        'kind': 'op',
        'value': np.array(4, dtype=np.int64),
        'op': 'Const',
        'type': 'Const'
    },
    'split1_axis_const_data': {
        'value': np.array(4, dtype=np.int64),
        'shape': np.array(4, dtype=np.int64).shape,
        'kind': 'data'
    },
    'split2': {'type': 'Split', 'kind': 'op', 'op': 'Split', 'num_splits': 2},
    'split2_axis_const': {
        'kind': 'op',
        'value': np.array(4, dtype=np.int64),
        'op': 'Const',
        'type': 'Const'
    },
    'split2_axis_const_data': {
        'value': np.array(4, dtype=np.int64),
        'shape': np.array(4, dtype=np.int64).shape,
        'kind': 'data'
    },
    'split1_data_0': {'value': None, 'shape': int64_array([1, 13, 13, 3, 1]), 'kind': 'data'},
    'split1_data_1': {'value': None, 'shape': int64_array([1, 13, 13, 3, 1]), 'kind': 'data'},
    'split2_data_0': {'value': None, 'shape': int64_array([1, 13, 13, 3, 1]), 'kind': 'data'},
    'split2_data_1': {'value': None, 'shape': int64_array([1, 13, 13, 3, 1]), 'kind': 'data'},
    'concat': {'type': 'Concat', 'kind': 'op', 'axis': 4},
    'concat_data': {'value': None, 'shape': int64_array([1, 13, 13, 3, 4]), 'kind': 'data'},
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'abs_data': {'value': None, 'shape': int64_array([1, 13, 13, 3, 4]), 'kind': 'data'},
    'output': {'kind': 'op', 'op': 'Result'},
}


graph_edges_when_there_are_two_splits_one_concat = [
    ('placeholder1', 'placeholder1_data'),
    ('placeholder2', 'placeholder2_data'),
    ('placeholder1_data', 'split1', {'in': 0}),
    ('split1_axis_const', 'split1_axis_const_data'),
    ('split1_axis_const_data', 'split1', {'in': 1}),
    ('split1', 'split1_data_0', {'out': 0}),
    ('split1', 'split1_data_1', {'out': 1}),
    ('placeholder2_data', 'split2', {'in': 0}),
    ('split2_axis_const', 'split2_axis_const_data'),
    ('split2_axis_const_data', 'split2', {'in': 1}),
    ('split2', 'split2_data_0', {'out': 0}),
    ('split2', 'split2_data_1', {'out': 1}),
    ('split1_data_0', 'concat', {'in': 0}),
    ('split1_data_1', 'concat', {'in': 1}),
    ('split2_data_0', 'concat', {'in': 2}),
    ('split2_data_1', 'concat', {'in': 3}),
    ('concat', 'concat_data'),
    ('concat_data', 'abs'),
    ('abs', 'abs_data'),
    ('abs_data', 'output')
]


class SplitConcatPairToInterpolateTest(unittest.TestCase):
    def test_spatial_2d_split_concat_1(self):
        graph = build_graph(
            nodes_attrs=graph_node_attrs_for_2d_spatial_case,
            edges=graph_edges
        )
        ref_graph = build_graph(
            nodes_attrs=ref_graph_node_attrs_for_2d_spatial_case_1_opset4,
            edges=ref_graph_edges_opset4
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
                'split_axis_const_data': {
                    'value': np.array(2, dtype=np.int64),
                    'shape': np.array(2, dtype=np.int64).shape,
                    'kind': 'data'
                },
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
            edges=ref_graph_edges_opset4,
            update_attributes={
                'axes': {'shape': int64_array([1]), 'value': int64_array([2])}
            }
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
            edges=ref_graph_edges_opset4,
            update_attributes={
                'axes': {'shape': int64_array([1]), 'value': int64_array([4])}
            }
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
                'split_axis_const_data': {
                    'value': np.array(3, dtype=np.int64),
                    'shape': np.array(3, dtype=np.int64).shape,
                    'kind': 'data'
                },
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
            edges=ref_graph_edges_opset4
        )
        SplitConcatPairToInterpolate().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output')
        self.assertTrue(flag, resp)

    def test_two_splits_one_concat(self):
        graph = build_graph(
            nodes_attrs=graph_node_attrs_when_there_are_two_splits_one_concat,
            edges=graph_edges_when_there_are_two_splits_one_concat
        )
        ref_graph = build_graph(
            nodes_attrs=graph_node_attrs_when_there_are_two_splits_one_concat,
            edges=graph_edges_when_there_are_two_splits_one_concat
        )
        SplitConcatPairToInterpolate().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output')
        self.assertTrue(flag, resp)
