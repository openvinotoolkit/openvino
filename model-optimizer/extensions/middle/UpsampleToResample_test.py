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
from generator import generator, generate

from extensions.middle.UpsampleToResample import UpsampleToResample
from mo.front.common.partial_infer.utils import int64_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph

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
    'cast_to_float': {'kind': 'op', 'op': 'Cast', 'type': 'Convert', 'dst_type': np.float},
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


@generator
class UpsampleToResampleTest(unittest.TestCase):
    @generate(*[([2, 10, 20, 30], [1, 1, 5, 5],),
                ([2, 20, 30, 40], [1, 1, 3, 3],),
                ([2, 3, 20, 30, 40], [1, 1, 3, 3, 3],)
                ])
    def test_conversion(self, input_shape, scales):
        graph = build_graph(graph_node_attrs, graph_edges,
                            {'placeholder_data': {'shape': int64_array(input_shape)},
                             'scales': {'value': int64_array(scales), 'shape': int64_array(scales).shape},
                             'scales_data': {'value': int64_array(scales), 'shape': int64_array(scales).shape},
                             'upsample_data': {'shape': int64_array(input_shape) * int64_array(scales)}})
        graph.graph['layout'] = 'NCHW'

        ref_graph = build_graph(ref_graph_node_attrs, ref_graph_edges,
                                {'placeholder_data': {'shape': int64_array(input_shape)},
                                 'factor': {'value': int64_array(scales)[2:], 'shape': int64_array(scales[2:]).shape},
                                 'interpolate_data': {'shape': int64_array(input_shape) * int64_array(scales)},
                                 'interpolate': {'axes': list(range(2, len(input_shape)))}}
                                )

        UpsampleToResample().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output')
        self.assertTrue(flag, resp)

    @generate(*[([2, 10, 20, 30], [1, 2, 5, 5],),
                ([2, 10, 20, 30], [1, 1, 6, 5],),
                ([2, 20, 30, 40], [1, 1, 3, 4],),
                ([2, 3, 20, 30, 40], [1, 1, 3, 4, 3],),
                ([2, 3, 20, 30, 40], [1, 1, 4, 3, 3],),
                ([2, 3, 20, 30, 40], [1, 1, 3, 3, 4],),
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
        self.assertTrue(flag, resp)
