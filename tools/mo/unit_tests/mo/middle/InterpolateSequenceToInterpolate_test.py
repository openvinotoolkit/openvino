# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import unittest

from openvino.tools.mo.middle.InterpolateSequenceToInterpolate import InterpolateSequenceToInterpolate
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph

graph_node_attrs_for_2d_case_1_opset4_case = {
    'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_data': {
        'value': None,
        'shape': int64_array([1, 4, 220, 350]),
        'kind': 'data',
        'data_type': None
    },
    'size_1': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([660])
    },
    'size_1_data': {'value': int64_array([660]), 'shape': [1], 'kind': 'data'},
    'scale_1': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': np.array([3.0])
    },
    'scale_1_data': {'value': np.array([3.0]), 'shape': [1], 'kind': 'data'},
    'axes_1': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([2])
    },
    'axes_1_data': {'value': int64_array([2]), 'shape': [1], 'kind': 'data'},
    'interpolate_1': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'mode': 'nearest',
        'shape_calculation_mode': 'scales',
        'version': 'opset4'
    },
    'interpolate_1_data': {'value': None, 'shape': int64_array([1, 4, 660, 350]), 'kind': 'data'},
    'size_2': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([700])
    },
    'size_2_data': {'value': int64_array([700]), 'shape': [1], 'kind': 'data'},
    'scale_2': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': np.array([2.0])
    },
    'scale_2_data': {'value': np.array([2.0]), 'shape': [1], 'kind': 'data'},
    'axes_2': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([3])
    },
    'axes_2_data': {'value': int64_array([3]), 'shape': [1], 'kind': 'data'},
    'interpolate_2': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'mode': 'nearest',
        'shape_calculation_mode': 'scales',
        'version': 'opset4'
    },
    'interpolate_2_data': {'value': None, 'shape': int64_array([1, 4, 660, 700]), 'kind': 'data'},
    'size_3': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([1320])
    },
    'size_3_data': {'value': int64_array([1320]), 'shape': [1], 'kind': 'data'},
    'scale_3': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': np.array([2.0])
    },
    'scale_3_data': {'value': np.array([2.0]), 'shape': [1], 'kind': 'data'},
    'axes_3': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([2])
    },
    'axes_3_data': {'value': int64_array([2]), 'shape': [1], 'kind': 'data'},
    'interpolate_3': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'mode': 'nearest',
        'shape_calculation_mode': 'scales',
        'version': 'opset4'
    },
    'interpolate_3_data': {'value': None, 'shape': int64_array([1, 4, 1320, 700]), 'kind': 'data'},
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'abs_data': {'value': None, 'shape': int64_array([1, 4, 1320, 700]), 'kind': 'data'},
    'output': {'kind': 'op', 'op': 'Result'},
}

edges_for_2d_case_1_opset4_case = [
    ('placeholder', 'placeholder_data'),

    ('placeholder_data', 'interpolate_1', {'in': 0}),
    ('size_1', 'size_1_data'),
    ('scale_1', 'scale_1_data'),
    ('axes_1', 'axes_1_data'),
    ('size_1_data', 'interpolate_1', {'in': 1}),
    ('scale_1_data', 'interpolate_1', {'in': 2}),
    ('axes_1_data', 'interpolate_1', {'in': 3}),
    ('interpolate_1', 'interpolate_1_data'),

    ('interpolate_1_data', 'interpolate_2', {'in': 0}),
    ('size_2', 'size_2_data'),
    ('scale_2', 'scale_2_data'),
    ('axes_2', 'axes_2_data'),
    ('size_2_data', 'interpolate_2', {'in': 1}),
    ('scale_2_data', 'interpolate_2', {'in': 2}),
    ('axes_2_data', 'interpolate_2', {'in': 3}),
    ('interpolate_2', 'interpolate_2_data'),

    ('interpolate_2_data', 'interpolate_3', {'in': 0}),
    ('size_3', 'size_3_data'),
    ('scale_3', 'scale_3_data'),
    ('axes_3', 'axes_3_data'),
    ('size_3_data', 'interpolate_3', {'in': 1}),
    ('scale_3_data', 'interpolate_3', {'in': 2}),
    ('axes_3_data', 'interpolate_3', {'in': 3}),
    ('interpolate_3', 'interpolate_3_data'),

    ('interpolate_3_data', 'abs'),
    ('abs', 'abs_data'),
    ('abs_data', 'output'),
]


ref_graph_node_attrs_for_2d_case_1_opset4_case = {
    'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_data': {
        'value': None,
        'shape': int64_array([1, 4, 220, 350]),
        'kind': 'data',
        'data_type': None
    },
    'size_1': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([660, 700])
    },
    'size_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    'scale_1': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': np.array([3.0, 2.0])
    },
    'scale_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    'axes_1': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([2, 3])
    },
    'axes_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    'interpolate_1': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'mode': 'nearest',
        'shape_calculation_mode': 'scales',
        'antialias': 0,
        'pads_begin': int64_array([0]),
        'pads_end': int64_array([0]),
        'coordinate_transformation_mode': 'half_pixel',
        'nearest_mode': 'round_prefer_floor',
        'cube_coeff': -0.75,
        'version': 'opset4'
    },
    'interpolate_1_data': {'value': None, 'shape': int64_array([1, 4, 660, 700]), 'kind': 'data'},
    'size_3': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([1320])
    },
    'size_3_data': {'value': None, 'shape': [1], 'kind': 'data'},
    'scale_3': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': np.array([2.0])
    },
    'scale_3_data': {'value': None, 'shape': [1], 'kind': 'data'},
    'axes_3': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([2])
    },
    'axes_3_data': {'value': int64_array([2]), 'shape': [1], 'kind': 'data'},
    'interpolate_3': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'mode': 'nearest',
        'shape_calculation_mode': 'scales',
        'version': 'opset4'
    },
    'interpolate_3_data': {'value': None, 'shape': int64_array([1, 4, 1320, 700]), 'kind': 'data'},
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'abs_data': {'value': None, 'shape': int64_array([1, 4, 1320, 700]), 'kind': 'data'},
    'output': {'kind': 'op', 'op': 'Result'},
}

ref_edges_for_2d_case_1_opset4_case = [
    ('placeholder', 'placeholder_data'),

    ('placeholder_data', 'interpolate_1', {'in': 0}),
    ('size_1', 'size_1_data'),
    ('scale_1', 'scale_1_data'),
    ('axes_1', 'axes_1_data'),
    ('size_1_data', 'interpolate_1', {'in': 1}),
    ('scale_1_data', 'interpolate_1', {'in': 2}),
    ('axes_1_data', 'interpolate_1', {'in': 3}),
    ('interpolate_1', 'interpolate_1_data'),

    ('interpolate_1_data', 'interpolate_3', {'in': 0}),
    ('size_3', 'size_3_data'),
    ('scale_3', 'scale_3_data'),
    ('axes_3', 'axes_3_data'),
    ('size_3_data', 'interpolate_3', {'in': 1}),
    ('scale_3_data', 'interpolate_3', {'in': 2}),
    ('axes_3_data', 'interpolate_3', {'in': 3}),
    ('interpolate_3', 'interpolate_3_data'),

    ('interpolate_3_data', 'abs'),
    ('abs', 'abs_data'),
    ('abs_data', 'output'),
]


graph_node_attrs_for_2d_case_1 = {
    'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_data': {
        'value': None,
        'shape': int64_array([1, 4, 220, 350]),
        'kind': 'data',
        'data_type': None
    },
    'scale_1': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([660])
    },
    'scale_1_data': {'value': int64_array([660]), 'shape': [1], 'kind': 'data'},
    'interpolate_1': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'axes': int64_array([2]),
        'mode': 'nearest',
        'version': 'opset1'
    },
    'interpolate_1_data': {'value': None, 'shape': int64_array([1, 4, 660, 350]), 'kind': 'data'},
    'scale_2': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([700])
    },
    'scale_2_data': {'value': int64_array([700]), 'shape': [1], 'kind': 'data'},
    'interpolate_2': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'axes': int64_array([3]),
        'mode': 'nearest',
        'version': 'opset1'
    },
    'interpolate_2_data': {'value': None, 'shape': int64_array([1, 4, 660, 700]), 'kind': 'data'},
    'scale_3': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([1320])
    },
    'scale_3_data': {'value': int64_array([1320]), 'shape': [1], 'kind': 'data'},
    'interpolate_3': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'axes': int64_array([2]),
        'mode': 'nearest',
        'version': 'opset1'
    },
    'interpolate_3_data': {'value': None, 'shape': int64_array([1, 4, 1320, 700]), 'kind': 'data'},
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'abs_data': {'value': None, 'shape': int64_array([1, 4, 1320, 700]), 'kind': 'data'},
    'output': {'kind': 'op', 'op': 'Result'},
}

edges_for_2d_case_1 = [
    ('placeholder', 'placeholder_data'),

    ('placeholder_data', 'interpolate_1', {'in': 0}),
    ('scale_1', 'scale_1_data'),
    ('scale_1_data', 'interpolate_1', {'in': 1}),
    ('interpolate_1', 'interpolate_1_data'),

    ('interpolate_1_data', 'interpolate_2', {'in': 0}),
    ('scale_2', 'scale_2_data'),
    ('scale_2_data', 'interpolate_2', {'in': 1}),
    ('interpolate_2', 'interpolate_2_data'),

    ('interpolate_2_data', 'interpolate_3', {'in': 0}),
    ('scale_3', 'scale_3_data'),
    ('scale_3_data', 'interpolate_3', {'in': 1}),
    ('interpolate_3', 'interpolate_3_data'),

    ('interpolate_3_data', 'abs'),
    ('abs', 'abs_data'),
    ('abs_data', 'output'),
]


graph_node_attrs_for_2d_case_2 = {
    'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_data': {
        'value': None,
        'shape': int64_array([1, 4, 220, 350]),
        'kind': 'data',
        'data_type': None
    },
    'scale_1': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([660])
    },
    'scale_1_data': {'value': None, 'shape': [1], 'kind': 'data'},
    'interpolate_1': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'axes': int64_array([2]),
        'mode': 'nearest',
        'version': 'opset1'
    },
    'interpolate_1_data': {'value': None, 'shape': int64_array([1, 4, 660, 350]), 'kind': 'data'},
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'abs_data': {'value': None, 'shape': int64_array([1, 4, 660, 350]), 'kind': 'data'},
    'output': {'kind': 'op', 'op': 'Result'},
}

edges_for_2d_case_2 = [
    ('placeholder', 'placeholder_data'),

    ('placeholder_data', 'interpolate_1', {'in': 0}),
    ('scale_1', 'scale_1_data'),
    ('scale_1_data', 'interpolate_1', {'in': 1}),
    ('interpolate_1', 'interpolate_1_data'),

    ('interpolate_1_data', 'abs'),
    ('abs', 'abs_data'),
    ('abs_data', 'output'),
]


graph_node_attrs_for_2d_case_3 = {
    'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_data': {
        'value': None,
        'shape': int64_array([1, 4, 220, 350]),
        'kind': 'data',
        'data_type': None
    },
    'scale_1': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([660])
    },
    'scale_1_data': {'value': None, 'shape': [1], 'kind': 'data'},
    'interpolate_1': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'axes': int64_array([2]),
        'mode': 'nearest',
        'version': 'opset1'
    },
    'interpolate_1_data': {'value': None, 'shape': int64_array([1, 4, 660, 350]), 'kind': 'data'},
    'scale_2': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([700])
    },
    'scale_2_data': {'value': None, 'shape': [1], 'kind': 'data'},
    'interpolate_2': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'axes': int64_array([3]),
        'mode': 'linear',
        'version': 'opset1'
    },
    'interpolate_2_data': {'value': None, 'shape': int64_array([1, 4, 660, 700]), 'kind': 'data'},
    'scale_3': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([1320])
    },
    'scale_3_data': {'value': None, 'shape': [1], 'kind': 'data'},
    'interpolate_3': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'axes': int64_array([2]),
        'mode': 'cubic',
        'version': 'opset1'
    },
    'interpolate_3_data': {'value': None, 'shape': int64_array([1, 4, 1320, 700]), 'kind': 'data'},
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'abs_data': {'value': None, 'shape': int64_array([1, 4, 1320, 700]), 'kind': 'data'},
    'output': {'kind': 'op', 'op': 'Result'},
}

edges_for_2d_case_3 = edges_for_2d_case_1


new_graph_node_attrs_for_2d_case_4_opset4_case = {
    'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_data': {
        'value': None,
        'shape': int64_array([1, 4, 220, 350]),
        'kind': 'data',
        'data_type': None
    },
    'size_1': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([2200])
    },
    'size_1_data': {'value': int64_array([2200]), 'shape': [1], 'kind': 'data'},
    'scale_1': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': np.array([10.0])
    },
    'scale_1_data': {'value': np.array([10.0]), 'shape': [1], 'kind': 'data'},
    'axes_1': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([2])
    },
    'axes_1_data': {'value': int64_array([2]), 'shape': [1], 'kind': 'data'},
    'interpolate_1': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'mode': 'linear',
        'coordinate_transformation_mode': 'asymmetric',
        'nearest_mode': 'simple',
        'cube_coeff': -0.4,
        'antialias': 1,
        'shape_calculation_mode': 'scales',
        'version': 'opset4'
    },
    'interpolate_1_data': {'value': None, 'shape': int64_array([1, 4, 2200, 350]), 'kind': 'data'},
    'size_2': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([700])
    },
    'size_2_data': {'value': int64_array([700]), 'shape': [1], 'kind': 'data'},
    'scale_2': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': np.array([2.0])
    },
    'scale_2_data': {'value': np.array([2.0]), 'shape': [1], 'kind': 'data'},
    'axes_2': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([3])
    },
    'axes_2_data': {'value': int64_array([3]), 'shape': [1], 'kind': 'data'},
    'interpolate_2': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'mode': 'linear',
        'coordinate_transformation_mode': 'asymmetric',
        'nearest_mode': 'simple',
        'cube_coeff': -0.4,
        'antialias': 1,
        'shape_calculation_mode': 'scales',
        'version': 'opset4'
    },
    'interpolate_2_data': {'value': None, 'shape': int64_array([1, 4, 2200, 700]), 'kind': 'data'},
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'abs_data': {'value': None, 'shape': int64_array([1, 4, 2200, 700]), 'kind': 'data'},
    'output': {'kind': 'op', 'op': 'Result'},
}

new_edges_for_2d_case_4_opset4_case = [
    ('placeholder', 'placeholder_data'),

    ('placeholder_data', 'interpolate_1', {'in': 0}),
    ('size_1', 'size_1_data'),
    ('size_1_data', 'interpolate_1', {'in': 1}),
    ('scale_1', 'scale_1_data'),
    ('scale_1_data', 'interpolate_1', {'in': 2}),
    ('axes_1', 'axes_1_data'),
    ('axes_1_data', 'interpolate_1', {'in': 3}),
    ('interpolate_1', 'interpolate_1_data'),

    ('interpolate_1_data', 'interpolate_2', {'in': 0}),
    ('size_2', 'size_2_data'),
    ('size_2_data', 'interpolate_2', {'in': 1}),
    ('scale_2', 'scale_2_data'),
    ('scale_2_data', 'interpolate_2', {'in': 2}),
    ('axes_2', 'axes_2_data'),
    ('axes_2_data', 'interpolate_2', {'in': 3}),
    ('interpolate_2', 'interpolate_2_data'),

    ('interpolate_2_data', 'abs'),
    ('abs', 'abs_data'),
    ('abs_data', 'output'),
]


new_ref_graph_node_attrs_for_2d_case_4_opset4_case = {
    'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_data': {
        'value': None,
        'shape': int64_array([1, 4, 220, 350]),
        'kind': 'data',
        'data_type': None
    },
    'size_1': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([2200, 700])
    },
    'size_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    'scale_1': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': np.array([10.0, 2.0])
    },
    'scale_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    'axes_1': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([2, 3])
    },
    'axes_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    'interpolate_1': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'mode': 'linear',
        'coordinate_transformation_mode': 'asymmetric',
        'nearest_mode': 'simple',
        'cube_coeff': -0.4,
        'antialias': 1,
        'shape_calculation_mode': 'scales',
        'version': 'opset4'
    },
    'interpolate_1_data': {'value': None, 'shape': int64_array([1, 4, 2200, 700]), 'kind': 'data'},
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'abs_data': {'value': None, 'shape': int64_array([1, 4, 2200, 700]), 'kind': 'data'},
    'output': {'kind': 'op', 'op': 'Result'},
}

new_ref_edges_for_2d_case_4_opset4_case = [
    ('placeholder', 'placeholder_data'),

    ('placeholder_data', 'interpolate_1', {'in': 0}),
    ('size_1', 'size_1_data'),
    ('size_1_data', 'interpolate_1', {'in': 1}),
    ('scale_1', 'scale_1_data'),
    ('scale_1_data', 'interpolate_1', {'in': 2}),
    ('axes_1', 'axes_1_data'),
    ('axes_1_data', 'interpolate_1', {'in': 3}),
    ('interpolate_1', 'interpolate_1_data'),

    ('interpolate_1_data', 'abs'),
    ('abs', 'abs_data'),
    ('abs_data', 'output'),
]


graph_node_attrs_for_2d_case_4_opset4_case = {
    'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_data': {
        'value': None,
        'shape': int64_array([1, 4, 220, 350]),
        'kind': 'data',
        'data_type': None
    },
    'scale_1': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([2200])
    },
    'scale_1_data': {'value': None, 'shape': [1], 'kind': 'data'},
    'axes_1': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([2])
    },
    'axes_1_data': {'value': int64_array([2]), 'shape': [1], 'kind': 'data'},
    'interpolate_1': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'mode': 'linear',
        'coordinate_transformation_mode': 'asymmetric',
        'nearest_mode': 'simple',
        'cube_coeff': -0.4,
        'antialias': 1,
        'version': 'opset4'
    },
    'interpolate_1_data': {'value': None, 'shape': int64_array([1, 4, 2200, 350]), 'kind': 'data'},
    'scale_2': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([700])
    },
    'scale_2_data': {'value': None, 'shape': [1], 'kind': 'data'},
    'axes_2': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([3])
    },
    'axes_2_data': {'value': int64_array([3]), 'shape': [1], 'kind': 'data'},
    'interpolate_2': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'mode': 'linear',
        'coordinate_transformation_mode': 'asymmetric',
        'nearest_mode': 'simple',
        'cube_coeff': -0.4,
        'antialias': 1,
        'version': 'opset4'
    },
    'interpolate_2_data': {'value': None, 'shape': int64_array([1, 4, 2200, 700]), 'kind': 'data'},
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'abs_data': {'value': None, 'shape': int64_array([1, 4, 2200, 700]), 'kind': 'data'},
    'output': {'kind': 'op', 'op': 'Result'},
}

edges_for_2d_case_4_opset4_case = [
    ('placeholder', 'placeholder_data'),

    ('placeholder_data', 'interpolate_1', {'in': 0}),
    ('scale_1', 'scale_1_data'),
    ('scale_1_data', 'interpolate_1', {'in': 1}),
    ('axes_1', 'axes_1_data'),
    ('axes_1_data', 'interpolate_1', {'in': 2}),
    ('interpolate_1', 'interpolate_1_data'),

    ('interpolate_1_data', 'interpolate_2', {'in': 0}),
    ('scale_2', 'scale_2_data'),
    ('scale_2_data', 'interpolate_2', {'in': 1}),
    ('axes_2', 'axes_2_data'),
    ('axes_2_data', 'interpolate_2', {'in': 2}),
    ('interpolate_2', 'interpolate_2_data'),

    ('interpolate_2_data', 'abs'),
    ('abs', 'abs_data'),
    ('abs_data', 'output'),
]


graph_node_attrs_for_2d_case_4 = {
    'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_data': {
        'value': None,
        'shape': int64_array([1, 4, 220, 350]),
        'kind': 'data',
        'data_type': None
    },
    'scale_1': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([2200])
    },
    'scale_1_data': {'value': int64_array([2200]), 'shape': [1], 'kind': 'data'},
    'interpolate_1': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'axes': int64_array([2]),
        'mode': 'linear',
        'align_corners': 0,
        'antialias': 1,
        'pads_begin': 5,
        'pads_end': 3,
        'version': 'opset1'
    },
    'interpolate_1_data': {'value': None, 'shape': int64_array([1, 4, 2200, 350]), 'kind': 'data'},
    'scale_2': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([700])
    },
    'scale_2_data': {'value': int64_array([700]), 'shape': [1], 'kind': 'data'},
    'interpolate_2': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'axes': int64_array([3]),
        'mode': 'linear',
        'align_corners': 0,
        'antialias': 1,
        'pads_begin': 5,
        'pads_end': 3,
        'version': 'opset1'
    },
    'interpolate_2_data': {'value': None, 'shape': int64_array([1, 4, 2200, 700]), 'kind': 'data'},
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'abs_data': {'value': None, 'shape': int64_array([1, 4, 2200, 700]), 'kind': 'data'},
    'output': {'kind': 'op', 'op': 'Result'},
}

edges_for_2d_case_4 = [
    ('placeholder', 'placeholder_data'),

    ('placeholder_data', 'interpolate_1', {'in': 0}),
    ('scale_1', 'scale_1_data'),
    ('scale_1_data', 'interpolate_1', {'in': 1}),
    ('interpolate_1', 'interpolate_1_data'),

    ('interpolate_1_data', 'interpolate_2', {'in': 0}),
    ('scale_2', 'scale_2_data'),
    ('scale_2_data', 'interpolate_2', {'in': 1}),
    ('interpolate_2', 'interpolate_2_data'),

    ('interpolate_2_data', 'abs'),
    ('abs', 'abs_data'),
    ('abs_data', 'output'),
]


graph_node_attrs_for_2d_case_6 = {
    'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_data': {
        'value': None,
        'shape': int64_array([1, 4, 220, 350]),
        'kind': 'data',
        'data_type': None
    },
    'scale_1': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([220, 350])
    },
    'scale_1_data': {'value': None, 'shape': [2], 'kind': 'data'},
    'interpolate_1': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'axes': int64_array([2, 3]),
        'mode': 'linear',
        'align_corners': 0,
        'antialias': 1,
        'pads_begin': 5,
        'pads_end': 3,
        'version': 'opset1'
    },
    'interpolate_1_data': {'value': None, 'shape': int64_array([1, 4, 220, 350]), 'kind': 'data'},
    'scale_2': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([220])
    },
    'scale_2_data': {'value': None, 'shape': [1], 'kind': 'data'},
    'interpolate_2': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'axes': int64_array([2]),
        'mode': 'linear',
        'align_corners': 0,
        'antialias': 1,
        'pads_begin': 5,
        'pads_end': 3,
        'version': 'opset1'
    },
    'interpolate_2_data': {'value': None, 'shape': int64_array([1, 4, 220, 350]), 'kind': 'data'},
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'abs_data': {'value': None, 'shape': int64_array([1, 4, 220, 350]), 'kind': 'data'},
    'output': {'kind': 'op', 'op': 'Result'},
}

edges_for_2d_case_6 = edges_for_2d_case_4


new_ref_graph_node_attrs_for_3d_case_1_opset4_case = {
    'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_data': {
        'value': None,
        'shape': int64_array([1, 5, 1024, 256, 800]),
        'kind': 'data',
        'data_type': None
    },
    'size_1': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([4096, 1280, 2400])
    },
    'size_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    'scale_1': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': np.array([4.0, 5.0, 3.0])
    },
    'scale_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    'axes_1': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([2, 3, 4])
    },
    'axes_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    'interpolate_1': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'mode': 'nearest',
        'shape_calculation_mode': 'sizes',
        'version': 'opset4'
    },
    'interpolate_1_data': {'value': None, 'shape': int64_array([1, 5, 4096, 1280, 2400]), 'kind': 'data'},
    'size_3': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([512])
    },
    'size_3_data': {'value': None, 'shape': [1], 'kind': 'data'},
    'scale_3': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': np.array([512.0 / 2400.0])
    },
    'scale_3_data': {'value': None, 'shape': [1], 'kind': 'data'},
    'axes_3': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([4])
    },
    'axes_3_data': {'value': int64_array([4]), 'shape': [1], 'kind': 'data'},
    'interpolate_3': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'mode': 'nearest',
        'shape_calculation_mode': 'sizes',
        'version': 'opset4'
    },
    'interpolate_3_data': {'value': None, 'shape': int64_array([1, 5, 4096, 1280, 512]), 'kind': 'data'},
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'abs_data': {'value': None, 'shape': int64_array([1, 5, 4096, 1280, 512]), 'kind': 'data'},
    'output': {'kind': 'op', 'op': 'Result'},
}


new_ref_edges_for_3d_case_1_opset4_case = ref_edges_for_2d_case_1_opset4_case


new_graph_node_attrs_for_3d_case_1_opset4_case = {
    'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_data': {
        'value': None,
        'shape': int64_array([1, 5, 1024, 256, 800]),
        'kind': 'data',
        'data_type': None
    },
    'size_1': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([4096, 2400])
    },
    'size_1_data': {'value': int64_array([4096, 2400]), 'shape': [2], 'kind': 'data'},
    'scale_1': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': np.array([4.0, 3.0])
    },
    'scale_1_data': {'value': np.array([4.0, 3.0]), 'shape': [2], 'kind': 'data'},
    'axes_1': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([2, 4])
    },
    'axes_1_data': {'value': int64_array([2, 4]), 'shape': [2], 'kind': 'data'},
    'interpolate_1': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'mode': 'nearest',
        'shape_calculation_mode': 'sizes',
        'version': 'opset4'
    },
    'interpolate_1_data': {'value': None, 'shape': int64_array([1, 5, 4096, 256, 2400]), 'kind': 'data'},
    'size_2': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([1280])
    },
    'size_2_data': {'value': int64_array([1280]), 'shape': [1], 'kind': 'data'},
    'scale_2': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': np.array([5.0])
    },
    'scale_2_data': {'value': np.array([5.0]), 'shape': [1], 'kind': 'data'},
    'axes_2': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([3])
    },
    'axes_2_data': {'value': int64_array([3]), 'shape': [1], 'kind': 'data'},
    'interpolate_2': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'mode': 'nearest',
        'shape_calculation_mode': 'sizes',
        'version': 'opset4'
    },
    'interpolate_2_data': {'value': None, 'shape': int64_array([1, 5, 4096, 1280, 2400]), 'kind': 'data'},
    'size_3': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([512])
    },
    'size_3_data': {'value': int64_array([512]), 'shape': [1], 'kind': 'data'},
    'scale_3': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': np.array([512.0 / 2400.0])
    },
    'scale_3_data': {'value': np.array([512.0 / 2400.0]), 'shape': [1], 'kind': 'data'},
    'axes_3': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([4])
    },
    'axes_3_data': {'value': int64_array([4]), 'shape': [1], 'kind': 'data'},
    'interpolate_3': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'mode': 'nearest',
        'shape_calculation_mode': 'sizes',
        'version': 'opset4'
    },
    'interpolate_3_data': {'value': None, 'shape': int64_array([1, 5, 4096, 1280, 512]), 'kind': 'data'},
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'abs_data': {'value': None, 'shape': int64_array([1, 5, 4096, 1280, 512]), 'kind': 'data'},
    'output': {'kind': 'op', 'op': 'Result'},
}

new_edges_for_3d_case_1_opset4_case = edges_for_2d_case_1_opset4_case


graph_node_attrs_for_3d_case_1 = {
    'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_data': {
        'value': None,
        'shape': int64_array([1, 5, 1024, 256, 800]),
        'kind': 'data',
        'data_type': None
    },
    'scale_1': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([4096, 2400])
    },
    'scale_1_data': {'value': int64_array([4096, 2400]), 'shape': [2], 'kind': 'data'},
    'interpolate_1': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'axes': int64_array([2, 4]),
        'mode': 'nearest',
        'version': 'opset1'
    },
    'interpolate_1_data': {'value': None, 'shape': int64_array([1, 5, 4096, 256, 2400]), 'kind': 'data'},
    'scale_2': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([1280])
    },
    'scale_2_data': {'value': int64_array([1280]), 'shape': [1], 'kind': 'data'},
    'interpolate_2': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'axes': int64_array([3]),
        'mode': 'nearest',
        'version': 'opset1'
    },
    'interpolate_2_data': {'value': None, 'shape': int64_array([1, 5, 4096, 1280, 2400]), 'kind': 'data'},
    'scale_3': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([512])
    },
    'scale_3_data': {'value': int64_array([512]), 'shape': [1], 'kind': 'data'},
    'interpolate_3': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'axes': int64_array([4]),
        'mode': 'nearest',
        'version': 'opset1'
    },
    'interpolate_3_data': {'value': None, 'shape': int64_array([1, 5, 4096, 1280, 512]), 'kind': 'data'},
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'abs_data': {'value': None, 'shape': int64_array([1, 5, 4096, 1280, 512]), 'kind': 'data'},
    'output': {'kind': 'op', 'op': 'Result'},
}

edges_for_3d_case_1 = edges_for_2d_case_1


graph_node_attrs_for_3d_case_2 = {
    'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_data': {
        'value': None,
        'shape': int64_array([1, 5, 1024, 256, 800]),
        'kind': 'data',
        'data_type': None
    },
    'scale_1': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([4096, 1280])
    },
    'scale_1_data': {'value': None, 'shape': [1], 'kind': 'data'},
    'interpolate_1': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'axes': int64_array([2, 3]),
        'mode': 'nearest',
        'version': 'opset1'
    },
    'interpolate_1_data': {'value': None, 'shape': int64_array([1, 5, 4096, 1280, 800]), 'kind': 'data'},
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'abs_data': {'value': None, 'shape': int64_array([1, 5, 4096, 1280, 800]), 'kind': 'data'},
    'output': {'kind': 'op', 'op': 'Result'},
}

edges_for_3d_case_2 = edges_for_2d_case_2


graph_node_attrs_for_3d_case_3 = {
    'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_data': {
        'value': None,
        'shape': int64_array([16, 44, 512, 87, 790]),
        'kind': 'data',
        'data_type': None
    },
    'scale_1': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([256])
    },
    'scale_1_data': {'value': None, 'shape': [1], 'kind': 'data'},
    'interpolate_1': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'axes': int64_array([2]),
        'mode': 'nearest',
        'version': 'opset1'
    },
    'interpolate_1_data': {'value': None, 'shape': int64_array([16, 44, 256, 87, 790]), 'kind': 'data'},
    'scale_2': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([2370])
    },
    'scale_2_data': {'value': None, 'shape': [1], 'kind': 'data'},
    'interpolate_2': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'axes': int64_array([4]),
        'mode': 'linear',
        'version': 'opset1'
    },
    'interpolate_2_data': {'value': None, 'shape': int64_array([16, 44, 256, 87, 2370]), 'kind': 'data'},
    'scale_3': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([435])
    },
    'scale_3_data': {'value': None, 'shape': [1], 'kind': 'data'},
    'interpolate_3': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'axes': int64_array([3]),
        'mode': 'cubic',
        'version': 'opset1'
    },
    'interpolate_3_data': {'value': None, 'shape': int64_array([16, 44, 256, 435, 2370]), 'kind': 'data'},
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'abs_data': {'value': None, 'shape': int64_array([16, 44, 256, 435, 2370]), 'kind': 'data'},
    'output': {'kind': 'op', 'op': 'Result'},
}

edges_for_3d_case_3 = edges_for_2d_case_3


new_ref_graph_node_attrs_for_3d_case_4_opset4_case = {
    'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_data': {
        'value': None,
        'shape': int64_array([10, 64, 511, 416, 10240]),
        'kind': 'data',
        'data_type': None
    },
    'size_1': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([4599, 912, 133120])
    },
    'size_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    'scale_1': {
        'kind': 'op', 'op': 'Const', 'type': 'Const',
        'value': np.array([4599.0 / 511.0, 912.0 / 416.0, 133120.0 / 10240.0])
    },
    'scale_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    'axes_1': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([2, 3, 4])
    },
    'axes_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    'interpolate_1': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'mode': 'linear',
        'antialias': 1,
        'shape_calculation_mode': 'sizes',
        'version': 'opset4'
    },
    'interpolate_1_data': {'value': None, 'shape': int64_array([10, 64, 4599, 912, 133120]), 'kind': 'data'},
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'abs_data': {'value': None, 'shape': int64_array([10, 64, 4599, 912, 133120]), 'kind': 'data'},
    'output': {'kind': 'op', 'op': 'Result'},
}

new_ref_edges_for_3d_case_4_opset4_case = new_ref_edges_for_2d_case_4_opset4_case


new_graph_node_attrs_for_3d_case_4_opset4_case = {
    'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_data': {
        'value': None,
        'shape': int64_array([10, 64, 511, 416, 10240]),
        'kind': 'data',
        'data_type': None
    },
    'size_1': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([4599, 133120])
    },
    'size_1_data': {'value': int64_array([4599, 133120]), 'shape': [2], 'kind': 'data'},
    'scale_1': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': np.array([4599.0 / 511.0, 133120.0 / 10240.0])
    },
    'scale_1_data': {'value': np.array([4599.0 / 511.0, 133120.0 / 10240.0]), 'shape': [2], 'kind': 'data'},
    'axes_1': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([2, 4])
    },
    'axes_1_data': {'value': int64_array([2, 4]), 'shape': [2], 'kind': 'data'},
    'interpolate_1': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'mode': 'linear',
        'antialias': 1,
        'shape_calculation_mode': 'sizes',
        'version': 'opset4'
    },
    'interpolate_1_data': {'value': None, 'shape': int64_array([10, 64, 4599, 416, 133120]), 'kind': 'data'},
    'size_2': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([912])
    },
    'size_2_data': {'value': int64_array([912]), 'shape': [1], 'kind': 'data'},
    'scale_2': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': np.array([912.0 / 416.0])
    },
    'scale_2_data': {'value': np.array([912.0 / 416.0]), 'shape': [1], 'kind': 'data'},
    'axes_2': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([3])
    },
    'axes_2_data': {'value': int64_array([3]), 'shape': [1], 'kind': 'data'},
    'interpolate_2': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'mode': 'linear',
        'antialias': 1,
        'shape_calculation_mode': 'sizes',
        'version': 'opset4'
    },
    'interpolate_2_data': {'value': None, 'shape': int64_array([10, 64, 4599, 912, 133120]), 'kind': 'data'},
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'abs_data': {'value': None, 'shape': int64_array([10, 64, 4599, 912, 133120]), 'kind': 'data'},
    'output': {'kind': 'op', 'op': 'Result'},
}

new_edges_for_3d_case_4_opset4_case = new_edges_for_2d_case_4_opset4_case


graph_node_attrs_for_3d_case_4 = {
    'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_data': {
        'value': None,
        'shape': int64_array([10, 64, 511, 416, 10240]),
        'kind': 'data',
        'data_type': None
    },
    'scale_1': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([4599, 133120])
    },
    'scale_1_data': {'value': int64_array([4599, 133120]), 'shape': [2], 'kind': 'data'},
    'interpolate_1': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'axes': int64_array([2, 4]),
        'mode': 'linear',
        'align_corners': 0,
        'antialias': 1,
        'pads_begin': 5,
        'pads_end': 3,
        'version': 'opset1'
    },
    'interpolate_1_data': {'value': None, 'shape': int64_array([10, 64, 4599, 416, 133120]), 'kind': 'data'},
    'scale_2': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([912])
    },
    'scale_2_data': {'value': int64_array([912]), 'shape': [1], 'kind': 'data'},
    'interpolate_2': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'axes': int64_array([3]),
        'mode': 'linear',
        'align_corners': 0,
        'antialias': 1,
        'pads_begin': 5,
        'pads_end': 3,
        'version': 'opset1'
    },
    'interpolate_2_data': {'value': None, 'shape': int64_array([10, 64, 4599, 912, 133120]), 'kind': 'data'},
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'abs_data': {'value': None, 'shape': int64_array([10, 64, 4599, 912, 133120]), 'kind': 'data'},
    'output': {'kind': 'op', 'op': 'Result'},
}

edges_for_3d_case_4 = edges_for_2d_case_4


class InterpolateSequenceToInterpolateTest(unittest.TestCase):
    def test_2d_interpolate_sequence_1(self):
        graph = build_graph(
            nodes_attrs=graph_node_attrs_for_2d_case_1,
            edges=edges_for_2d_case_1
        )

        ref_graph = build_graph(
            nodes_attrs={
                'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
                'placeholder_data': {
                    'value': None,
                    'shape': int64_array([1, 4, 220, 350]),
                    'kind': 'data',
                    'data_type': None
                },
                'scale_1': {
                    'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([660, 700])
                },
                'scale_1_data': {'value': None, 'shape': None, 'kind': 'data'},
                'interpolate_1': {
                    'type': 'Interpolate',
                    'kind': 'op',
                    'op': 'Interpolate',
                    'axes': int64_array([2, 3]),
                    'mode': 'nearest',
                    'version': 'opset1'
                },
                'interpolate_1_data': {'value': None, 'shape': int64_array([1, 4, 660, 700]), 'kind': 'data'},
                'scale_2': {
                    'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([1320])
                },
                'scale_2_data': {'value': None, 'shape': None, 'kind': 'data'},
                'interpolate_2': {
                    'type': 'Interpolate',
                    'kind': 'op',
                    'op': 'Interpolate',
                    'axes': int64_array([2]),
                    'mode': 'nearest',
                    'version': 'opset1'
                },
                'interpolate_2_data': {'value': None, 'shape': int64_array([1, 4, 1320, 700]), 'kind': 'data'},
                'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
                'abs_data': {'value': None, 'shape': int64_array([1, 4, 1320, 700]), 'kind': 'data'},
                'output': {'kind': 'op', 'op': 'Result'},
            },
            edges=[
                ('placeholder', 'placeholder_data'),
                ('placeholder_data', 'interpolate_1', {'in': 0}),
                ('scale_1', 'scale_1_data'),
                ('scale_1_data', 'interpolate_1', {'in': 1}),
                ('interpolate_1', 'interpolate_1_data'),
                ('scale_2', 'scale_2_data'),
                ('interpolate_2', 'interpolate_2_data'),
                ('interpolate_1_data', 'interpolate_2', {'in': 0}),
                ('scale_2_data', 'interpolate_2', {'in': 1}),
                ('interpolate_2_data', 'abs'),
                ('abs', 'abs_data'),
                ('abs_data', 'output'),
            ]
        )
        InterpolateSequenceToInterpolate().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output')
        self.assertTrue(flag, resp)

    def test_2d_interpolate_sequence_1_opset4_case(self):
        graph = build_graph(
            nodes_attrs=graph_node_attrs_for_2d_case_1_opset4_case,
            edges=edges_for_2d_case_1_opset4_case
        )

        ref_graph = build_graph(
            nodes_attrs=ref_graph_node_attrs_for_2d_case_1_opset4_case,
            edges=ref_edges_for_2d_case_1_opset4_case
        )
        InterpolateSequenceToInterpolate().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output')
        self.assertTrue(flag, resp)

    def test_2d_interpolate_sequence_2(self):
        graph = build_graph(
            nodes_attrs=graph_node_attrs_for_2d_case_2,
            edges=edges_for_2d_case_2
        )
        ref_graph = build_graph(
            nodes_attrs=graph_node_attrs_for_2d_case_2,
            edges=edges_for_2d_case_2
        )
        InterpolateSequenceToInterpolate().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output')
        self.assertTrue(flag, resp)

    def test_2d_interpolate_sequence_3(self):
        graph = build_graph(
            nodes_attrs=graph_node_attrs_for_2d_case_3,
            edges=edges_for_2d_case_3
        )

        ref_graph = build_graph(
            nodes_attrs=graph_node_attrs_for_2d_case_3,
            edges=edges_for_2d_case_3
        )

        InterpolateSequenceToInterpolate().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output')
        self.assertTrue(flag, resp)

    def test_2d_interpolate_sequence_4(self):
        graph = build_graph(
            nodes_attrs=graph_node_attrs_for_2d_case_4,
            edges=edges_for_2d_case_4
        )

        ref_graph = build_graph(
            nodes_attrs={
                'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
                'placeholder_data': {
                    'value': None,
                    'shape': int64_array([1, 4, 220, 350]),
                    'kind': 'data',
                    'data_type': None
                },
                'scale': {
                    'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([2200, 700])
                },
                'scale_data': {'value': None, 'shape': None, 'kind': 'data'},
                'interpolate': {
                    'type': 'Interpolate',
                    'kind': 'op',
                    'op': 'Interpolate',
                    'axes': int64_array([2, 3]),
                    'mode': 'linear',
                    'align_corners': 0,
                    'antialias': 1,
                    'pads_begin': 5,
                    'pads_end': 3,
                    'version': 'opset1'
                },
                'interpolate_data': {'value': None, 'shape': int64_array([1, 4, 2200, 700]), 'kind': 'data'},
                'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
                'abs_data': {'value': None, 'shape': int64_array([1, 4, 2200, 700]), 'kind': 'data'},
                'output': {'kind': 'op', 'op': 'Result'},
            },
            edges=[
                ('placeholder', 'placeholder_data'),

                ('placeholder_data', 'interpolate', {'in': 0}),
                ('scale', 'scale_data'),
                ('scale_data', 'interpolate', {'in': 1}),
                ('interpolate', 'interpolate_data'),

                ('interpolate_data', 'abs'),
                ('abs', 'abs_data'),
                ('abs_data', 'output'),
            ]
        )

        InterpolateSequenceToInterpolate().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output')
        self.assertTrue(flag, resp)

    def test_2d_interpolate_sequence_4_opset4_case(self):
        graph = build_graph(
            nodes_attrs=new_graph_node_attrs_for_2d_case_4_opset4_case,
            edges=new_edges_for_2d_case_4_opset4_case
        )

        ref_graph = build_graph(
            nodes_attrs=new_ref_graph_node_attrs_for_2d_case_4_opset4_case,
            edges=new_ref_edges_for_2d_case_4_opset4_case
        )

        InterpolateSequenceToInterpolate().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output')
        self.assertTrue(flag, resp)

    def test_2d_interpolate_sequence_5(self):
        graph = build_graph(
            nodes_attrs=graph_node_attrs_for_2d_case_4,
            edges=edges_for_2d_case_4,
            update_attributes={
                'interpolate_1': {
                    'align_corners': 1, 'antialias': 1, 'pads_begin': 3, 'pads_end': 0
                }
            }
        )

        ref_graph = build_graph(
            nodes_attrs=graph_node_attrs_for_2d_case_4,
            edges=edges_for_2d_case_4,
            update_attributes={
                'interpolate_1': {
                    'align_corners': 1, 'antialias': 1, 'pads_begin': 3, 'pads_end': 0
                }
            }
        )

        InterpolateSequenceToInterpolate().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output')
        self.assertTrue(flag, resp)

    def test_2d_interpolate_sequence_5_opset4_case(self):
        graph = build_graph(
            nodes_attrs=graph_node_attrs_for_2d_case_4_opset4_case,
            edges=edges_for_2d_case_4_opset4_case,
            update_attributes={
                'interpolate_1': {
                    'antialias': 0, 'cube_coeff': -0.1
                }
            }
        )

        ref_graph = build_graph(
            nodes_attrs=graph_node_attrs_for_2d_case_4_opset4_case,
            edges=edges_for_2d_case_4_opset4_case,
            update_attributes={
                'interpolate_1': {
                    'antialias': 0, 'cube_coeff': -0.1
                }
            }
        )

        InterpolateSequenceToInterpolate().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output')
        self.assertTrue(flag, resp)

    def test_2d_interpolate_sequence_6(self):
        graph = build_graph(
            nodes_attrs=graph_node_attrs_for_2d_case_6,
            edges=edges_for_2d_case_6,
        )

        ref_graph = build_graph(
            nodes_attrs=graph_node_attrs_for_2d_case_6,
            edges=edges_for_2d_case_6
        )

        InterpolateSequenceToInterpolate().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output')
        self.assertTrue(flag, resp)

    def test_3d_interpolate_sequence_1(self):
        graph = build_graph(
            nodes_attrs=graph_node_attrs_for_3d_case_1,
            edges=edges_for_3d_case_1
        )

        ref_graph = build_graph(
            nodes_attrs={
                'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
                'placeholder_data': {
                    'value': None,
                    'shape': int64_array([1, 5, 1024, 256, 800]),
                    'kind': 'data',
                    'data_type': None
                },
                'scale_1': {
                    'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([4096, 1280, 2400])
                },
                'scale_1_data': {'value': None, 'shape': None, 'kind': 'data'},
                'interpolate_1': {
                    'type': 'Interpolate',
                    'kind': 'op',
                    'op': 'Interpolate',
                    'axes': int64_array([2, 3, 4]),
                    'mode': 'nearest',
                    'version': 'opset1'
                },
                'interpolate_1_data': {'value': None, 'shape': int64_array([1, 5, 4096, 1280, 2400]), 'kind': 'data'},
                'scale_2': {
                    'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([512])
                },
                'scale_2_data': {'value': None, 'shape': [1], 'kind': 'data'},
                'interpolate_2': {
                    'type': 'Interpolate',
                    'kind': 'op',
                    'op': 'Interpolate',
                    'axes': int64_array([4]),
                    'mode': 'nearest',
                    'version': 'opset1'
                },
                'interpolate_2_data': {'value': None, 'shape': int64_array([1, 5, 4096, 1280, 512]), 'kind': 'data'},
                'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
                'abs_data': {'value': None, 'shape': int64_array([1, 5, 4096, 1280, 512]), 'kind': 'data'},
                'output': {'kind': 'op', 'op': 'Result'},
            },
            edges=[
                ('placeholder', 'placeholder_data'),
                ('placeholder_data', 'interpolate_1', {'in': 0}),
                ('scale_1', 'scale_1_data'),
                ('scale_1_data', 'interpolate_1', {'in': 1}),
                ('interpolate_1', 'interpolate_1_data'),
                ('scale_2', 'scale_2_data'),
                ('interpolate_2', 'interpolate_2_data'),
                ('interpolate_1_data', 'interpolate_2', {'in': 0}),
                ('scale_2_data', 'interpolate_2', {'in': 1}),
                ('interpolate_2_data', 'abs'),
                ('abs', 'abs_data'),
                ('abs_data', 'output'),
            ]
        )
        InterpolateSequenceToInterpolate().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output')
        self.assertTrue(flag, resp)

    def test_3d_interpolate_sequence_1_opset4_case(self):
        graph = build_graph(
            nodes_attrs=new_graph_node_attrs_for_3d_case_1_opset4_case,
            edges=new_edges_for_3d_case_1_opset4_case
        )

        ref_graph = build_graph(
            nodes_attrs=new_ref_graph_node_attrs_for_3d_case_1_opset4_case,
            edges=new_ref_edges_for_3d_case_1_opset4_case
        )
        InterpolateSequenceToInterpolate().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output')
        self.assertTrue(flag, resp)

    def test_3d_interpolate_sequence_2(self):
        graph = build_graph(
            nodes_attrs=graph_node_attrs_for_3d_case_2,
            edges=edges_for_3d_case_2
        )
        ref_graph = build_graph(
            nodes_attrs=graph_node_attrs_for_3d_case_2,
            edges=edges_for_3d_case_2
        )
        InterpolateSequenceToInterpolate().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output')
        self.assertTrue(flag, resp)

    def test_3d_interpolate_sequence_3(self):
        graph = build_graph(
            nodes_attrs=graph_node_attrs_for_3d_case_3,
            edges=edges_for_3d_case_3
        )
        ref_graph = build_graph(
            nodes_attrs=graph_node_attrs_for_3d_case_3,
            edges=edges_for_3d_case_3
        )
        InterpolateSequenceToInterpolate().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output')
        self.assertTrue(flag, resp)

    def test_3d_interpolate_sequence_4(self):
        graph = build_graph(
            nodes_attrs=graph_node_attrs_for_3d_case_4,
            edges=edges_for_3d_case_4
        )

        ref_graph = build_graph(
            nodes_attrs={
                'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
                'placeholder_data': {
                    'value': None,
                    'shape': int64_array([10, 64, 511, 416, 10240]),
                    'kind': 'data',
                    'data_type': None
                },
                'scale': {
                    'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([4599, 912, 133120])
                },
                'scale_data': {'value': None, 'shape': None, 'kind': 'data'},
                'interpolate': {
                    'type': 'Interpolate',
                    'kind': 'op',
                    'op': 'Interpolate',
                    'axes': int64_array([2, 3, 4]),
                    'mode': 'linear',
                    'align_corners': 0,
                    'antialias': 1,
                    'pads_begin': 5,
                    'pads_end': 3,
                    'version': 'opset1'
                },
                'interpolate_data': {'value': None, 'shape': int64_array([10, 64, 4599, 912, 133120]), 'kind': 'data'},
                'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
                'abs_data': {'value': None, 'shape': int64_array([10, 64, 4599, 912, 133120]), 'kind': 'data'},
                'output': {'kind': 'op', 'op': 'Result'},
            },
            edges=[
                ('placeholder', 'placeholder_data'),

                ('placeholder_data', 'interpolate', {'in': 0}),
                ('scale', 'scale_data'),
                ('scale_data', 'interpolate', {'in': 1}),
                ('interpolate', 'interpolate_data'),

                ('interpolate_data', 'abs'),
                ('abs', 'abs_data'),
                ('abs_data', 'output'),
            ]
        )

        InterpolateSequenceToInterpolate().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output')
        self.assertTrue(flag, resp)

    def test_3d_interpolate_sequence_4_opset4_case(self):
        graph = build_graph(
            nodes_attrs=new_graph_node_attrs_for_3d_case_4_opset4_case,
            edges=new_edges_for_3d_case_4_opset4_case
        )

        ref_graph = build_graph(
            nodes_attrs=new_ref_graph_node_attrs_for_3d_case_4_opset4_case,
            edges=new_ref_edges_for_3d_case_4_opset4_case
        )

        InterpolateSequenceToInterpolate().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output')
        self.assertTrue(flag, resp)

    def test_3d_interpolate_sequence_5(self):
        graph = build_graph(
            nodes_attrs=graph_node_attrs_for_3d_case_4,
            edges=edges_for_3d_case_4,
            update_attributes={
                'interpolate_1': {
                    'align_corners': 1, 'antialias': 1, 'pads_begin': 3, 'pads_end': 7
                }
            }
        )

        ref_graph = build_graph(
            nodes_attrs=graph_node_attrs_for_3d_case_4,
            edges=edges_for_3d_case_4,
            update_attributes={
                'interpolate_1': {
                    'align_corners': 1, 'antialias': 1, 'pads_begin': 3, 'pads_end': 7
                }
            }
        )

        InterpolateSequenceToInterpolate().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output')
        self.assertTrue(flag, resp)
