# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

import numpy as np

from openvino.tools.mo.middle.UpsampleToResample import UpsampleToResample
from openvino.tools.mo.front.common.partial_infer.utils import int64_array, float32_array
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph

graph_node_attrs = {
    'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': np.float32},
    'scales': {'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': None, 'shape': None},
    'scales_data': {'kind': 'data', 'value': None, 'shape': None},
    'upsample': {'type': None, 'kind': 'op', 'op': 'Upsample', 'mode': 'linear'},
    'upsample_data': {'kind': 'data', 'shape': None, 'value': None},
    'output': {'kind': 'op', 'op': 'Result', 'type': 'Result'},
}

graph_edges = [
    ('placeholder', 'placeholder_data'),
    ('placeholder_data', 'upsample', {'in': 0}),
    ('scales', 'scales_data'),
    ('scales_data', 'upsample', {'in': 1}),
    ('upsample', 'upsample_data'),
    ('upsample_data', 'output'),
]

new_ref_graph_node_attr = {
    'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': np.float32},
    'ss_begin': {'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([2]), 'shape': int64_array([1])},
    'ss_begin_data': {'kind': 'data', 'value': int64_array([2]), 'shape': int64_array([1])},
    'ss_end': {'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([4]), 'shape': int64_array([1])},
    'ss_end_data': {'kind': 'data', 'value': int64_array([4]), 'shape': int64_array([1])},
    'ss_stride': {'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([1]), 'shape': int64_array([1])},
    'ss_stride_data': {'kind': 'data', 'value': int64_array([1]), 'shape': int64_array([1])},
    'strided_slice': {'type': 'StridedSlice', 'kind': 'op', 'op': 'StridedSlice'},
    'strided_slice_data': {'kind': 'data', 'shape': None, 'value': None},
    'cast_to_float': {'kind': 'op', 'op': 'Cast', 'type': 'Convert', 'dst_type': np.float32},
    'cast_to_float_d': {'kind': 'data', 'value': None, 'shape': None},
    'factor': {'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([5, 5]), 'shape': int64_array([2])},
    'factor_data': {'kind': 'data', 'value': int64_array([5, 5]), 'shape': int64_array([2])},
    'shapeof': {'type': 'ShapeOf', 'kind': 'op', 'op': 'ShapeOf'},
    'shapeof_data': {'kind': 'data', 'shape': None, 'value': None},
    'mul': {'type': 'Multiply', 'kind': 'op', 'op': 'Multiply'},
    'mul_data': {'kind': 'data', 'shape': None, 'value': None},
    'cast_to_int': {'kind': 'op', 'op': 'Cast', 'type': 'Convert', 'dst_type': np.int32},
    'cast_to_int_d': {'kind': 'data', 'shape': None, 'value': None},
    'axes_const': {'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': None, 'shape': None},
    'axes_const_data': {'kind': 'data', 'value': None, 'shape': None},
    'scales': {'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([5, 5]), 'shape': int64_array([2])},
    'scales_data': {'kind': 'data', 'value': None, 'shape': None},
    'interpolate': {'type': 'Interpolate', 'kind': 'op', 'op': 'Interpolate', 'axes': None},
    'interpolate_data': {'kind': 'data', 'shape': None, 'value': None},
    'output': {'kind': 'op', 'op': 'Result', 'type': 'Result'},
}

new_ref_graph_edges = [
    ('placeholder', 'placeholder_data'),
    ('placeholder_data', 'shapeof', {'in': 0, 'out': 0}),
    ('placeholder_data', 'interpolate', {'in': 0, 'out': 0}),
    ('ss_begin', 'ss_begin_data'),
    ('ss_begin_data', 'strided_slice', {'in': 1, 'out': 0}),
    ('ss_end', 'ss_end_data'),
    ('ss_end_data', 'strided_slice', {'in': 2, 'out': 0}),
    ('ss_stride', 'ss_stride_data'),
    ('ss_stride_data', 'strided_slice', {'in': 3, 'out': 0}),
    ('strided_slice', 'strided_slice_data'),
    ('strided_slice_data', 'cast_to_float'),
    ('cast_to_float', 'cast_to_float_d'),
    ('shapeof', 'shapeof_data'),
    ('shapeof_data', 'strided_slice', {'in': 0, 'out': 0}),
    ('factor', 'factor_data'),
    ('cast_to_float_d', 'mul', {'in': 0, 'out': 0}),
    ('factor_data', 'mul', {'in': 1, 'out': 0}),
    ('mul', 'mul_data'),
    ('mul_data', 'cast_to_int'),
    ('cast_to_int', 'cast_to_int_d'),
    ('cast_to_int_d', 'interpolate', {'in': 1, 'out': 0}),
    ('axes_const', 'axes_const_data'),
    ('axes_const_data', 'interpolate', {'in': 3, 'out': 0}),
    ('scales', 'scales_data'),
    ('scales_data', 'interpolate', {'in': 2, 'out': 0}),
    ('interpolate', 'interpolate_data'),
    ('interpolate_data', 'output')
]

ref_graph_node_attrs = {
    'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': np.float32},
    'factor': {'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([5, 5]), 'shape': int64_array([2])},
    'factor_data': {'kind': 'data', 'value': None, 'shape': None},
    'shapeof': {'type': 'ShapeOf', 'kind': 'op', 'op': 'ShapeOf'},
    'shapeof_data': {'kind': 'data', 'shape': None, 'value': None},
    'strided_slice': {'type': 'StridedSlice', 'kind': 'op', 'op': 'StridedSlice'},
    'strided_slice_data': {'kind': 'data', 'shape': None, 'value': None},
    'ss_begin': {'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([2]), 'shape': int64_array([1])},
    'ss_begin_data': {'kind': 'data', 'value': None, 'shape': None},
    'ss_end': {'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([4]), 'shape': int64_array([1])},
    'ss_end_data': {'kind': 'data', 'value': None, 'shape': None},
    'ss_stride': {'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([1]), 'shape': int64_array([1])},
    'ss_stride_data': {'kind': 'data', 'value': None, 'shape': None},
    'cast_to_float': {'kind': 'op', 'op': 'Cast', 'type': 'Convert', 'dst_type': np.float32},
    'cast_to_float_d': {'kind': 'data', 'value': None, 'shape': None},
    'mul': {'type': 'Multiply', 'kind': 'op', 'op': 'Multiply'},
    'mul_data': {'kind': 'data', 'shape': None, 'value': None},
    'cast_to_int': {'kind': 'op', 'op': 'Cast', 'type': 'Convert', 'dst_type': np.int32},
    'cast_to_int_d': {'kind': 'data', 'shape': None, 'value': None},
    'interpolate': {'type': 'Interpolate', 'kind': 'op', 'op': 'Interpolate', 'axes': None},
    'interpolate_data': {'kind': 'data', 'shape': None, 'value': None},
    'output': {'kind': 'op', 'op': 'Result', 'type': 'Result'},
}

ref_graph_edges = [
    ('placeholder', 'placeholder_data'),
    ('placeholder_data', 'interpolate', {'in': 0, 'out': 0}),
    ('placeholder_data', 'shapeof', {'in': 0, 'out': 0}),
    ('shapeof', 'shapeof_data'),
    ('interpolate', 'interpolate_data'),
    ('factor', 'factor_data'),
    ('shapeof_data', 'strided_slice', {'in': 0, 'out': 0}),
    ('ss_begin', 'ss_begin_data'),
    ('ss_begin_data', 'strided_slice', {'in': 1, 'out': 0}),
    ('ss_end', 'ss_end_data'),
    ('ss_end_data', 'strided_slice', {'in': 2, 'out': 0}),
    ('ss_stride', 'ss_stride_data'),
    ('ss_stride_data', 'strided_slice', {'in': 3, 'out': 0}),
    ('strided_slice', 'strided_slice_data'),
    ('strided_slice_data', 'cast_to_float'),
    ('cast_to_float', 'cast_to_float_d'),
    ('cast_to_float_d', 'mul', {'in': 0, 'out': 0}),
    ('factor_data', 'mul', {'in': 1, 'out': 0}),
    ('mul', 'mul_data'),
    ('mul_data', 'cast_to_int'),
    ('cast_to_int', 'cast_to_int_d'),
    ('cast_to_int_d', 'interpolate', {'in': 1, 'out': 0}),
    ('interpolate_data', 'output'),
]


class TestUpsampleToResampleTest():
    @pytest.mark.parametrize("input_shape, scales, axes",[([2, 10, 20, 30], [1, 1, 5, 5], [2, 3]),
                ([2, 20, 30, 40], [1, 1, 3, 3], [2, 3]),
                ([2, 10, 20, 30], [1, 1, 6, 5], [2, 3]),
                ([2, 20, 30, 40], [1, 1, 3, 4], [2, 3]),
                ([2, 3, 20, 30, 40], [1, 1, 3, 3, 3], [2, 3, 4]),
                ([2, 3, 20, 30, 40], [1, 1, 3, 4, 3], [2, 3, 4]),
                ([2, 3, 20, 30, 40], [1, 1, 4, 3, 3], [2, 3, 4]),
                ([2, 3, 20, 30, 40], [1, 1, 3, 3, 4], [2, 3, 4]),
                ([2, 10, 20, 30], [1, 1, 5.5, 5.7], [2, 3]),
                ([2, 20, 30, 40], [1, 1, 3.3, 3.1], [2, 3]),
                ([2, 10, 20, 30], [1, 1, 6.18, 5.34], [2, 3]),
                ([2, 20, 30, 40], [1, 1, 3.79, 4.16], [2, 3]),
                ([2, 3, 20, 30, 40], [1, 1, 3.12, 3.87, 3.92], [2, 3, 4]),
                ([2, 3, 20, 30, 40], [1, 1, 3.74, 4.873, 3.287], [2, 3, 4]),
                ([2, 3, 20, 30, 40], [1, 1, 4.8, 3.6, 3.11], [2, 3, 4]),
                ([2, 3, 20, 30, 40], [1, 1, 3.33, 3.73, 4.765], [2, 3, 4]),
                ])
    def test_conversion(self, input_shape, scales, axes):
        input_shape_as_array = int64_array(input_shape)
        scales_as_array = float32_array(scales)
        graph = build_graph(graph_node_attrs,
                            graph_edges,
                            {
                                'placeholder_data': {'shape': input_shape_as_array},
                                'scales': {'value': scales_as_array, 'shape': scales_as_array.shape},
                                'scales_data': {'value': scales_as_array, 'shape': scales_as_array.shape},
                                'upsample_data':
                                    {'shape': ((input_shape_as_array + 1.e-5) * scales_as_array).astype(np.int64)}
                            })
        graph.graph['layout'] = 'NCHW'
        ref_graph = build_graph(new_ref_graph_node_attr,
                                new_ref_graph_edges,
                                {
                                    'placeholder_data': {'shape': int64_array(input_shape)},
                                    'ss_begin': {'value': int64_array([axes[0]])},
                                    'ss_end': {'value': int64_array([axes[-1] + 1])},
                                    'ss_begin_data': {'value': int64_array([axes[0]])},
                                    'ss_end_data': {'value': int64_array([axes[-1] + 1])},
                                    'factor': {'value': scales_as_array[2:],
                                               'shape': scales_as_array[2:].shape},
                                    'factor_data': {'value': scales_as_array[2:],
                                                    'shape': scales_as_array[2:].shape},
                                    'axes_const': {'value': int64_array(axes), 'shape': int64_array(axes).shape},
                                    'interpolate_data': {
                                        'shape': (input_shape_as_array * scales_as_array + 1e-5).astype(np.int64)},
                                })
        UpsampleToResample().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output')
        assert flag, resp

    @pytest.mark.parametrize("input_shape, scales",[([2, 10, 20, 30], [1, 2, 5, 5],),
                ([2, 3, 20, 30, 40], [1, 2, 3, 3, 3],),
                ])
    def test_pattern_does_not_satisfy(self, input_shape, scales):
        graph = build_graph(graph_node_attrs, graph_edges,
                            {'placeholder_data': {'shape': int64_array(input_shape)},
                             'scales': {'value': int64_array(scales), 'shape': int64_array(scales).shape},
                             'scales_data': {'value': int64_array(scales), 'shape': int64_array(scales).shape},
                             'upsample_data': {'shape': int64_array(input_shape) * int64_array(scales)}})
        graph.graph['layout'] = 'NCHW'

        ref_graph = build_graph(graph_node_attrs, graph_edges,
                            {'placeholder_data': {'shape': int64_array(input_shape)},
                             'scales': {'value': int64_array(scales), 'shape': int64_array(scales).shape},
                             'scales_data': {'value': int64_array(scales), 'shape': int64_array(scales).shape},
                             'upsample_data': {'shape': int64_array(input_shape) * int64_array(scales)}})

        UpsampleToResample().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output')
        assert flag, resp
