# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

import numpy as np

from openvino.tools.mo.ops.interpolate import Interpolate
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.graph.graph import Node
from unit_tests.utils.graph import build_graph


graph_node_attrs_without_axes = {
    'input': {'type': 'Parameter', 'kind': 'op'},
    'input_data': {'kind': 'data', 'shape': None, 'value': None},
    'sizes': {'type': 'Const', 'kind': 'op', 'shape': None, 'value': None},
    'sizes_data': {'kind': 'data', 'shape': None, 'value': None},
    'scales': {'type': 'Const', 'kind': 'op', 'shape': None, 'value': None},
    'scales_data': {'kind': 'data', 'shape': None, 'value': None},
    'interpolate': {
        'type': 'Interpolate', 'kind': 'op', 'mode': 'nearest', 'shape_calculation_mode': 'sizes',
        'coordinate_transformation_mode': 'half_pixel', 'version': 'opset4',
        'nearest_mode': 'round_prefer_floor', 'antialias': 0,
    },
    'interpolate_data': {'kind': 'data', 'value': None, 'shape': None},
    'op_output': {'kind': 'op', 'op': 'Result'},
}

graph_edges_without_axes = [
    ('input', 'input_data'),
    ('sizes', 'sizes_data'),
    ('scales', 'scales_data'),
    ('input_data', 'interpolate', {'in': 0}),
    ('sizes_data', 'interpolate', {'in': 1}),
    ('scales_data', 'interpolate', {'in': 2}),
    ('interpolate', 'interpolate_data'),
    ('interpolate_data', 'op_output'),
]


graph_nodes_attrs = {
    'input': {'type': 'Parameter', 'kind': 'op'},
    'input_data': {'kind': 'data', 'shape': None, 'value': None},
    'sizes': {'type': 'Const', 'kind': 'op', 'shape': None, 'value': None},
    'sizes_data': {'kind': 'data', 'shape': None, 'value': None},
    'scales': {'type': 'Const', 'kind': 'op', 'shape': None, 'value': None},
    'scales_data': {'kind': 'data', 'shape': None, 'value': None},
    'axes': {'type': 'Const', 'kind': 'op', 'shape': None, 'value': None},
    'axes_data': {'kind': 'data', 'shape': None, 'value': None},
    'interpolate': {
        'type': 'Interpolate', 'kind': 'op', 'mode': 'nearest', 'shape_calculation_mode': 'sizes',
        'coordinate_transformation_mode': 'half_pixel', 'version': 'opset4',
        'nearest_mode': 'round_prefer_floor', 'antialias': 0,
    },
    'interpolate_data': {'kind': 'data', 'value': None, 'shape': None},
    'op_output': {'kind': 'op', 'op': 'Result'},
}

graph_edges = [
    ('input', 'input_data'),
    ('sizes', 'sizes_data'),
    ('scales', 'scales_data'),
    ('axes', 'axes_data'),
    ('input_data', 'interpolate', {'in': 0}),
    ('sizes_data', 'interpolate', {'in': 1}),
    ('scales_data', 'interpolate', {'in': 2}),
    ('axes_data', 'interpolate', {'in': 3}),
    ('interpolate', 'interpolate_data'),
    ('interpolate_data', 'op_output'),
]


class TestInterpolateOp():
    @pytest.mark.parametrize("pads_begin, pads_end, input_shape, output_shape, sizes, scales, axes",
                [([0], [0], [1, 3, 100, 200], [1, 3, 350, 150], [350, 150], [3.5, 150 / 200], [2, 3]),
                ([0, 3, 10, 10], [0], [16, 7, 190, 400], [8, 10, 390, 600],
                 [8, 390, 600], [0.5, 390 / 200, 600 / 410], [0, 2, 3]),
                ([10, 5, 0, 10], [0, 4, 16, 18], [4, 33, 1024, 8000], [56, 42, 520, 8028],
                 [56, 520], [4.0, 0.5], [0, 2]),
                ([0], [0], [1, 16, 85, 470, 690], [20, 16, 40, 470, 1380],
                 [20, 40, 1380], [20.0, 40.0 / 85.0, 1380.0 / 690.0], [0, 2, 4]),
                ([4, 3, 11, 22, 5], [1, 3, 4, 8, 5], [1, 16, 85, 470, 690], [60, 22, 430, 500, 345],
                 [60, 430, 345], [10.0, 4.3, 345.0 / 700.0], [0, 2, 4]),
                ([0], [0], [5, 77, 444, 88, 6050], [100, 308, 4440, 44, 6050],
                 [100, 308, 4440, 44], [20.0, 4.0, 10.0, 0.5], [0, 1, 2, 3]),
                ([0], [0], [1, 100, 200], [1, 350, 150], [350, 150], [3.5, 150 / 200], [1, 2]),
                ([0, 3, 10], [0], [16, 7, 190], [8, 10, 390], [8, 390], [0.5, 390 / 200], [0, 2]),
                ([10, 0, 10], [0, 16, 18], [4, 1024, 8000], [56, 520, 8028], [56, 520], [4.0, 0.5], [0, 1]),
                ([0], [0], [1, 690], [20, 1380], [20, 1380], [20.0, 1380.0 / 690.0], [0, 1]),
                ([4, 3, 11, 22, 5, 0], [1, 3, 4, 8, 5, 0], [1, 16, 85, 470, 690, 349], [60, 22, 430, 500, 345, 349],
                 [60, 430, 345], [10.0, 4.3, 345.0 / 700.0], [0, 2, 4])
                ])
    def test_interpolate4_using_sizes(self, pads_begin, pads_end, input_shape, output_shape, sizes, scales, axes):
        graph = build_graph(nodes_attrs=graph_nodes_attrs,
                            edges=graph_edges,
                            update_attributes={
                                'input_data': {'shape': input_shape},
                                'sizes': {'shape': int64_array(sizes).shape, 'value': int64_array(sizes)},
                                'sizes_data': {'shape': int64_array(sizes).shape, 'value': int64_array(sizes)},
                                'scales': {'shape': np.array(scales).shape, 'value': np.array(scales)},
                                'scales_data': {'shape': np.array(scales).shape, 'value': np.array(scales)},
                                'axes': {'shape': int64_array(axes).shape, 'value': int64_array(axes)},
                                'axes_data': {'shape': int64_array(axes).shape, 'value': int64_array(axes)},
                                'interpolate': {'pads_begin': int64_array(pads_begin),
                                                'pads_end': int64_array(pads_end)}
                            })

        node = Node(graph, 'interpolate')
        tested_class = Interpolate(graph=graph, attrs=node.attrs())
        tested_class.infer(node)

        msg = "Interpolate-4 infer failed for case: sizes={}, scales={}, pads_begin={}, pads_end={}, axes={}," \
              " expected_shape={}, actual_shape={}"

        assert np.array_equal(graph.node['interpolate_data']['shape'], int64_array(output_shape)),\
                        msg.format(sizes, scales, pads_begin, pads_end, axes, output_shape,
                                   graph.node['interpolate_data']['shape'])

    @pytest.mark.parametrize("pads_begin, pads_end, input_shape, output_shape, sizes, scales, axes",
                [([0], [0], [1, 3, 100, 200], [1, 3, 350, 150], [350, 150], [3.5, 150 / 200], [2, 3]),
                ([0, 3, 10, 10], [0], [16, 7, 190, 400], [8, 10, 390, 600],
                 [8, 390, 600], [0.5, 390 / 200, 600 / 410], [0, 2, 3]),
                ([10, 5, 0, 10], [0, 4, 16, 18], [4, 33, 1024, 8000], [56, 42, 520, 8028],
                 [56, 520], [4.0, 0.5], [0, 2]),
                ([0], [0], [1, 16, 85, 470, 690], [20, 16, 40, 470, 1380],
                 [20, 40, 1380], [20.0, 40.0 / 85.0, 1380.0 / 690.0], [0, 2, 4]),
                ([4, 3, 11, 22, 5], [1, 3, 4, 8, 5], [1, 16, 85, 470, 690], [60, 22, 430, 500, 345],
                 [60, 430, 345], [10.0, 4.3, 345.0 / 700.0], [0, 2, 4]),
                ([0], [0], [5, 77, 444, 88, 6050], [100, 308, 4440, 44, 6050],
                 [100, 308, 4440, 44], [20.0, 4.0, 10.0, 0.5], [0, 1, 2, 3]),
                ([0], [0], [1, 100, 200], [1, 350, 150], [350, 150], [3.5, 150 / 200], [1, 2]),
                ([0, 3, 10], [0], [16, 7, 190], [8, 10, 390], [8, 390], [0.5, 390 / 200], [0, 2]),
                ([10, 0, 10], [0, 16, 18], [4, 1024, 8000], [56, 520, 8028], [56, 520], [4.0, 0.5], [0, 1]),
                ([0], [0], [1, 690], [20, 1380], [20, 1380], [20.0, 1380.0 / 690.0], [0, 1]),
                ([4, 3, 11, 22, 5, 0], [1, 3, 4, 8, 5, 0], [1, 16, 85, 470, 690, 349], [60, 22, 430, 500, 345, 349],
                 [60, 430, 345], [10.0, 4.3, 345.0 / 700.0], [0, 2, 4]),
                ([4, 3, 11, 22, 5, 0, 0], [1, 3, 4, 8, 5, 0, 0], [1, 16, 85, 470, 690, 349, 3],
                 [60, 22, 430, 500, 345, 349, 1],
                 [60, 430, 345, 1], [10.0, 4.3, 345.0 / 700.0, 1 / 3], [0, 2, 4, 6]),
                ([4, 3, 11, 22, 5, 0, 0], [1, 3, 4, 8, 5, 0, 0], [1, 16, 85, 470, 690, 349, 3],
                 [60, 22, 430, 500, 345, 349, 1],
                 [60, 430, 345, 1], [10.0, 4.3, 345.0 / 700.0, 0.3333333], [0, 2, 4, 6]),
                ])
    def test_interpolate4_using_scales(self, pads_begin, pads_end, input_shape, output_shape, sizes, scales, axes):
        graph = build_graph(nodes_attrs=graph_nodes_attrs,
                            edges=graph_edges,
                            update_attributes={
                                'input_data': {'shape': input_shape},
                                'sizes': {'shape': int64_array(sizes).shape, 'value': int64_array(sizes)},
                                'sizes_data': {'shape': int64_array(sizes).shape, 'value': int64_array(sizes)},
                                'scales': {'shape': np.array(scales).shape, 'value': np.array(scales)},
                                'scales_data': {'shape': np.array(scales).shape, 'value': np.array(scales)},
                                'axes': {'shape': int64_array(axes).shape, 'value': int64_array(axes)},
                                'axes_data': {'shape': int64_array(axes).shape, 'value': int64_array(axes)},
                                'interpolate': {'pads_begin': int64_array(pads_begin),
                                                'pads_end': int64_array(pads_end),
                                                'shape_calculation_mode': 'scales'}
                            })

        node = Node(graph, 'interpolate')
        tested_class = Interpolate(graph=graph, attrs=node.attrs())
        tested_class.infer(node)

        msg = "Interpolate-4 infer failed for case: sizes={}, scales={}, pads_begin={}, pads_end={}, axes={}," \
              " expected_shape={}, actual_shape={}"

        assert np.array_equal(graph.node['interpolate_data']['shape'], int64_array(output_shape)),\
                        msg.format(sizes, scales, pads_begin, pads_end, axes, output_shape,
                                   graph.node['interpolate_data']['shape'])

    @pytest.mark.parametrize("pads_begin, pads_end, input_shape, output_shape, sizes, scales",
                [([0], [0], [1, 3, 100, 200], [1, 3, 350, 150], [1, 3, 350, 150], [1.0, 1.0, 3.5, 150 / 200]),
                ([0, 3, 10, 10], [0], [16, 7, 190, 400], [8, 10, 390, 600],
                 [8, 10, 390, 600], [0.5, 1.0, 390 / 200, 600 / 410]),
                ([10, 5, 0, 10], [0, 4, 16, 18], [4, 33, 1024, 8000], [56, 42, 520, 8028],
                 [56, 42, 520, 8028], [4.0, 1.0, 0.5, 1.0]),
                ([0], [0], [1, 16, 85, 470, 690], [20, 16, 40, 470, 1380],
                 [20, 16, 40, 470, 1380], [20.0, 1.0, 40.0 / 85.0, 1.0, 1380.0 / 690.0]),
                ([4, 3, 11, 22, 5], [1, 3, 4, 8, 5], [1, 16, 85, 470, 690], [60, 22, 430, 500, 345],
                 [60, 22, 430, 500, 345], [10.0, 1.0, 4.3, 1.0, 345.0 / 700.0]),
                ([0], [0], [5, 77, 444, 88, 6050], [100, 308, 4440, 44, 6050],
                 [100, 308, 4440, 44, 6050], [20.0, 4.0, 10.0, 0.5, 1.0]),
                ([0], [0], [1, 100, 200], [1, 350, 150], [1, 350, 150], [1.0, 3.5, 150 / 200]),
                ([0, 3, 10], [0], [16, 7, 190], [8, 10, 390], [8, 10, 390], [0.5, 1.0, 390 / 200]),
                ([10, 0, 10], [0, 16, 18], [4, 1024, 8000], [56, 520, 8028], [56, 520, 8028], [4.0, 0.5, 1.0]),
                ([0], [0], [1, 690], [20, 1380], [20, 1380], [20.0, 1380.0 / 690.0]),
                ([4, 3, 11, 22, 5, 0], [1, 3, 4, 8, 5, 0], [1, 16, 85, 470, 690, 349], [60, 22, 430, 500, 345, 349],
                 [60, 22, 430, 500, 345, 349], [10.0, 1.0, 4.3, 1.0, 345.0 / 700.0, 1.0]),
                ([4, 3, 11, 22, 5, 0, 0], [1, 3, 4, 8, 5, 0, 0], [1, 16, 85, 470, 690, 349, 3],
                 [60, 22, 430, 500, 345, 349, 1],
                 [60, 22, 430, 500, 345, 349, 1], [10.0, 1.0, 4.3, 1.0, 345.0 / 700.0, 1.0, 1 / 3]),
                ])
    def test_interpolate4_using_sizes_without_axes(self, pads_begin, pads_end, input_shape, output_shape, sizes,
                                                   scales):
        graph = build_graph(nodes_attrs=graph_node_attrs_without_axes,
                            edges=graph_edges_without_axes,
                            update_attributes={
                                'input_data': {'shape': input_shape},
                                'sizes': {'shape': int64_array(sizes).shape, 'value': int64_array(sizes)},
                                'sizes_data': {'shape': int64_array(sizes).shape, 'value': int64_array(sizes)},
                                'scales': {'shape': np.array(scales).shape, 'value': np.array(scales)},
                                'scales_data': {'shape': np.array(scales).shape, 'value': np.array(scales)},
                                'interpolate': {'pads_begin': int64_array(pads_begin),
                                                'pads_end': int64_array(pads_end),
                                                'shape_calculation_mode': 'sizes'}
                            })

        node = Node(graph, 'interpolate')
        tested_class = Interpolate(graph=graph, attrs=node.attrs())
        tested_class.infer(node)

        msg = "Interpolate-4 infer failed for case: sizes={}, scales={}, pads_begin={}, pads_end={}," \
              " expected_shape={}, actual_shape={}"

        assert np.array_equal(graph.node['interpolate_data']['shape'], int64_array(output_shape)),\
                        msg.format(sizes, scales, pads_begin, pads_end, output_shape,
                                   graph.node['interpolate_data']['shape'])

    @pytest.mark.parametrize("pads_begin, pads_end, input_shape, output_shape, sizes, scales",
                [([0], [0], [1, 3, 100, 200], [1, 3, 350, 150], [1, 3, 350, 150], [1.0, 1.0, 3.5, 150 / 200]),
                ([0, 3, 10, 10], [0], [16, 7, 190, 400], [8, 10, 390, 600],
                 [8, 10, 390, 600], [0.5, 1.0, 390 / 200, 600 / 410]),
                ([10, 5, 0, 10], [0, 4, 16, 18], [4, 33, 1024, 8000], [56, 42, 520, 8028],
                 [56, 42, 520, 8028], [4.0, 1.0, 0.5, 1.0]),
                ([0], [0], [1, 16, 85, 470, 690], [20, 16, 40, 470, 1380],
                 [20, 16, 40, 470, 1380], [20.0, 1.0, 40.0 / 85.0, 1.0, 1380.0 / 690.0]),
                ([4, 3, 11, 22, 5], [1, 3, 4, 8, 5], [1, 16, 85, 470, 690], [60, 22, 430, 500, 345],
                 [60, 22, 430, 500, 345], [10.0, 1.0, 4.3, 1.0, 345.0 / 700.0]),
                ([0], [0], [5, 77, 444, 88, 6050], [100, 308, 4440, 44, 6050],
                 [100, 308, 4440, 44, 6050], [20.0, 4.0, 10.0, 0.5, 1.0]),
                ([0], [0], [1, 100, 200], [1, 350, 150], [1, 350, 150], [1.0, 3.5, 150 / 200]),
                ([0, 3, 10], [0], [16, 7, 190], [8, 10, 390], [8, 10, 390], [0.5, 1.0, 390 / 200]),
                ([10, 0, 10], [0, 16, 18], [4, 1024, 8000], [56, 520, 8028], [56, 520, 8028], [4.0, 0.5, 1.0]),
                ([0], [0], [1, 690], [20, 1380], [20, 1380], [20.0, 1380.0 / 690.0]),
                ([4, 3, 11, 22, 5, 0], [1, 3, 4, 8, 5, 0], [1, 16, 85, 470, 690, 349], [60, 22, 430, 500, 345, 349],
                 [60, 22, 430, 500, 345, 349], [10.0, 1.0, 4.3, 1.0, 345.0 / 700.0, 1.0]),
                ([4, 3, 11, 22, 5, 0, 0], [1, 3, 4, 8, 5, 0, 0], [1, 16, 85, 470, 690, 349, 3],
                 [60, 22, 430, 500, 345, 349, 1],
                 [60, 22, 430, 500, 345, 349, 1], [10.0, 1.0, 4.3, 1.0, 345.0 / 700.0, 1.0, 1 / 3]),
                ([4, 3, 11, 22, 5, 0, 0], [1, 3, 4, 8, 5, 0, 0], [1, 16, 85, 470, 690, 349, 3],
                 [60, 22, 430, 500, 345, 349, 1],
                 [60, 22, 430, 500, 345, 349, 1], [10.0, 1.0, 4.3, 1.0, 345.0 / 700.0, 1.0, 0.3333333]),
                ])
    def test_interpolate4_using_scales_without_axes(self, pads_begin, pads_end, input_shape, output_shape, sizes,
                                                   scales):
        graph = build_graph(nodes_attrs=graph_node_attrs_without_axes,
                            edges=graph_edges_without_axes,
                            update_attributes={
                                'input_data': {'shape': input_shape},
                                'sizes': {'shape': int64_array(sizes).shape, 'value': int64_array(sizes)},
                                'sizes_data': {'shape': int64_array(sizes).shape, 'value': int64_array(sizes)},
                                'scales': {'shape': np.array(scales).shape, 'value': np.array(scales)},
                                'scales_data': {'shape': np.array(scales).shape, 'value': np.array(scales)},
                                'interpolate': {'pads_begin': int64_array(pads_begin),
                                                'pads_end': int64_array(pads_end),
                                                'shape_calculation_mode': 'scales'}
                            })

        node = Node(graph, 'interpolate')
        tested_class = Interpolate(graph=graph, attrs=node.attrs())
        tested_class.infer(node)

        msg = "Interpolate-4 infer failed for case: sizes={}, scales={}, pads_begin={}, pads_end={}," \
              " expected_shape={}, actual_shape={}"

        assert np.array_equal(graph.node['interpolate_data']['shape'], int64_array(output_shape)),\
                        msg.format(sizes, scales, pads_begin, pads_end, output_shape,
                                   graph.node['interpolate_data']['shape'])
