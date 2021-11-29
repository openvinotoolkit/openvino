# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
from generator import generator, generate

from extensions.ops.ONNXResize11 import ONNXResize11Op
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node
from unit_tests.utils.graph import build_graph


graph_node_attrs_sizes = {
    'input': {'type': 'Parameter', 'kind': 'op'},
    'input_data': {'kind': 'data', 'shape': None, 'value': None},
    'roi': {'type': 'Const', 'kind': 'op', 'shape': None, 'value': None},
    'roi_data': {'kind': 'data', 'shape': None, 'value': None},
    'scales': {'type': 'Const', 'kind': 'op', 'shape': None, 'value': None},
    'scales_data': {'kind': 'data', 'shape': None, 'value': None},
    'sizes': {'type': 'Const', 'kind': 'op', 'shape': None, 'value': None},
    'sizes_data': {'kind': 'data', 'shape': None, 'value': None},
    'onnx_resize11': {
        'op': 'ONNXResize11', 'kind': 'op', 'mode': 'nearest', 'nearest_mode': 'round_prefer_floor',
        'coordinate_transformation_mode': 'half_pixel', 'cube_coeff': -0.75
    },
    'onnx_resize11_data': {'kind': 'data', 'value': None, 'shape': None},
    'op_output': {'kind': 'op', 'op': 'Result'},
}

graph_edges_sizes = [
    ('input', 'input_data'),
    ('roi', 'roi_data'),
    ('sizes', 'sizes_data'),
    ('input_data', 'onnx_resize11', {'in': 0}),
    ('roi_data', 'onnx_resize11', {'in': 1}),
    ('sizes_data', 'onnx_resize11', {'in': 3}),
    ('onnx_resize11', 'onnx_resize11_data'),
    ('onnx_resize11_data', 'op_output'),
]


graph_node_attrs_scales = {
    'input': {'type': 'Parameter', 'kind': 'op'},
    'input_data': {'kind': 'data', 'shape': None, 'value': None},
    'roi': {'type': 'Const', 'kind': 'op', 'shape': None, 'value': None},
    'roi_data': {'kind': 'data', 'shape': None, 'value': None},
    'scales': {'type': 'Const', 'kind': 'op', 'shape': None, 'value': None},
    'scales_data': {'kind': 'data', 'shape': None, 'value': None},
    'onnx_resize11': {
        'op': 'ONNXResize11', 'kind': 'op', 'mode': 'nearest', 'nearest_mode': 'round_prefer_floor',
        'coordinate_transformation_mode': 'half_pixel', 'cube_coeff': -0.75
    },
    'onnx_resize11_data': {'kind': 'data', 'value': None, 'shape': None},
    'op_output': {'kind': 'op', 'op': 'Result'},
}

graph_edges_scales = [
    ('input', 'input_data'),
    ('roi', 'roi_data'),
    ('scales', 'scales_data'),
    ('input_data', 'onnx_resize11', {'in': 0}),
    ('roi_data', 'onnx_resize11', {'in': 1}),
    ('scales_data', 'onnx_resize11', {'in': 2}),
    ('onnx_resize11', 'onnx_resize11_data'),
    ('onnx_resize11_data', 'op_output'),
]


@generator
class TestONNXResize11Op(unittest.TestCase):
    @generate(*[([1, 260, 100, 150], [1, 260, 200, 350], [1, 260, 200, 350], [1.0, 1.0, 1.0, 1.0]),
                ([1, 260, 100, 150], [1, 260, 200, 350], [1, 1, 200, 350], [1.0, 1.0, 1.0, 1.0]),
                ([5, 14, 300, 40], [5, 14, 140, 280], [1, 1, 140, 280], [1.0, 1.0, 1.0, 1.0]),
                ([5, 14, 300, 40], [5, 14, 140, 280], [5, 14, 140, 280], [1.0, 1.0, 1.0, 1.0]),
                ([1, 3, 260, 100, 150], [1, 3, 780, 200, 350], [1, 3, 780, 200, 350], [1.0, 1.0, 1.0, 1.0, 1.0]),
                ([1, 3, 450, 100, 150], [1, 3, 260, 200, 350], [1, 3, 260, 200, 350], [1.0, 1.0, 1.0, 1.0, 1.0]),
                ([5, 14, 1000, 300, 40], [5, 14, 500, 140, 280], [1, 1, 500, 140, 280], [1.0, 1.0, 1.0, 1.0, 1.0]),
                ([5, 14, 1000, 300, 40], [5, 14, 500, 140, 280], [5, 14, 500, 140, 280], [1.0, 1.0, 1.0, 1.0, 1.0])])
    def test_onnx_resize11_using_sizes(self, input_shape, output_shape, sizes, scales):
        np_scales = np.array(scales)
        np_sizes = int64_array(sizes)
        graph = build_graph(nodes_attrs=graph_node_attrs_sizes,
                            edges=graph_edges_sizes,
                            update_attributes={
                                'input_data': {'shape': int64_array(input_shape)},
                                'scales': {'shape': int64_array(np_scales.shape), 'value': np_scales},
                                'scales_data': {'shape': int64_array(np_scales.shape), 'value': np_scales},
                                'sizes': {'shape': int64_array(np_sizes.shape), 'value': np_sizes},
                                'sizes_data': {'shape': int64_array(np_sizes.shape), 'value': np_sizes},
                            })
        node = Node(graph, 'onnx_resize11')
        ONNXResize11Op.onnx_resize_infer(node)

        msg = "ONNXResize11 infer failed for case: sizes={}, scales={}, expected_shape={}, actual_shape={}"

        self.assertTrue(np.array_equal(graph.node['onnx_resize11_data']['shape'], int64_array(output_shape)),
                        msg.format(sizes, scales, output_shape, graph.node['onnx_resize11_data']['shape']))

    @generate(*[([1, 260, 100, 150], [1, 260, 200, 350], [1.0, 1.0, 2.0, 350 / 150]),
                ([1, 3, 100, 200], [1, 3, 350, 150], [1.0, 1.0, 3.5, 150 / 200]),
                ([5, 14, 300, 40], [5, 14, 140, 280], [1.0, 1.0, 140 / 300, 7.0]),
                ([5, 14, 300, 40], [5, 14, 140, 560], [1.0, 1.0, 140 / 300, 14.0]),
                ([1, 3, 260, 100, 150], [1, 3, 780, 200, 350], [1.0, 1.0, 3.0, 2.0, 350 / 150]),
                ([1, 3, 450, 100, 150], [1, 3, 260, 200, 350], [1.0, 1.0, 260 / 450, 2.0, 350 / 150]),
                ([5, 14, 1000, 300, 40], [5, 14, 500, 140, 280], [1.0, 1.0, 0.5, 140 / 300, 7.0]),
                ([4, 3, 180, 1340], [4, 3, 60, 804], [1.0, 1.0, 0.33333334, 0.6]),
                ([4, 3, 500, 180, 1340], [4, 3, 750, 60, 804], [1.0, 1.0, 1.5, 0.33333334, 0.6])])
    def test_onnx_resize_using_scales(self, input_shape, output_shape, scales):
        np_scales = np.array(scales)
        graph = build_graph(nodes_attrs=graph_node_attrs_scales,
                            edges=graph_edges_scales,
                            update_attributes={
                                'input_data': {'shape': int64_array(input_shape)},
                                'scales': {'shape': int64_array(np_scales.shape), 'value': np_scales},
                                'scales_data': {'shape': int64_array(np_scales.shape), 'value': np_scales},
                            })
        node = Node(graph, 'onnx_resize11')
        ONNXResize11Op.onnx_resize_infer(node)

        msg = "ONNXResize11 infer failed for case: scales={}, expected_shape={}, actual_shape={}"

        self.assertTrue(np.array_equal(graph.node['onnx_resize11_data']['shape'], int64_array(output_shape)),
                        msg.format(scales, output_shape, graph.node['onnx_resize11_data']['shape']))

    @generate(*[([1, 260, 100, 150], [1, 260, 200, 350], [1, 260, 200, 350], [1.0, 1.0, 1.0, 1.0]),
                ([1, 260, 100, 150], [1, 260, 200, 350], [1, 1, 200, 350], [1.0, 1.0, 1.0, 1.0]),
                ([5, 14, 300, 40], [5, 14, 140, 280], [1, 1, 140, 280], [1.0, 1.0, 1.0, 1.0]),
                ([5, 14, 300, 40], [5, 14, 140, 280], [5, 14, 140, 280], [1.0, 1.0, 1.0, 1.0]),
                ([1, 3, 260, 100, 150], [1, 3, 780, 200, 350], [1, 3, 780, 200, 350], [1.0, 1.0, 1.0, 1.0, 1.0]),
                ([1, 3, 450, 100, 150], [1, 3, 260, 200, 350], [1, 3, 260, 200, 350], [1.0, 1.0, 1.0, 1.0, 1.0]),
                ([5, 14, 1000, 300, 40], [5, 14, 500, 140, 280], [1, 1, 500, 140, 280], [1.0, 1.0, 1.0, 1.0, 1.0]),
                ([5, 14, 1000, 300, 40], [5, 14, 500, 140, 280], [5, 14, 500, 140, 280], [1.0, 1.0, 1.0, 1.0, 1.0])])
    def test_onnx_resize11_using_sizes_without_roi_input(self, input_shape, output_shape, sizes, scales):
        np_scales = np.array(scales)
        np_sizes = int64_array(sizes)
        graph = build_graph(nodes_attrs=graph_node_attrs_sizes,
                            edges=[('input', 'input_data'),
                                   ('sizes', 'sizes_data'),
                                   ('input_data', 'onnx_resize11', {'in': 0}),
                                   ('sizes_data', 'onnx_resize11', {'in': 3}),
                                   ('onnx_resize11', 'onnx_resize11_data'),
                                   ('onnx_resize11_data', 'op_output'),
                                ],
                            update_attributes={
                                'input_data': {'shape': int64_array(input_shape)},
                                'scales': {'shape': int64_array(np_scales.shape), 'value': np_scales},
                                'scales_data': {'shape': int64_array(np_scales.shape), 'value': np_scales},
                                'sizes': {'shape': int64_array(np_sizes.shape), 'value': np_sizes},
                                'sizes_data': {'shape': int64_array(np_sizes.shape), 'value': np_sizes},
                            })
        node = Node(graph, 'onnx_resize11')
        ONNXResize11Op.onnx_resize_infer(node)

        msg = "ONNXResize11 infer failed for case: sizes={}, scales={}, expected_shape={}, actual_shape={}"

        self.assertTrue(np.array_equal(graph.node['onnx_resize11_data']['shape'], int64_array(output_shape)),
                        msg.format(sizes, scales, output_shape, graph.node['onnx_resize11_data']['shape']))

    @generate(*[([1, 260, 100, 150], [1, 260, 200, 350], [1.0, 1.0, 2.0, 350 / 150]),
                ([1, 3, 100, 200], [1, 3, 350, 150], [1.0, 1.0, 3.5, 150 / 200]),
                ([5, 14, 300, 40], [5, 14, 140, 280], [1.0, 1.0, 140 / 300, 7.0]),
                ([5, 14, 300, 40], [5, 14, 140, 560], [1.0, 1.0, 140 / 300, 14.0]),
                ([1, 3, 260, 100, 150], [1, 3, 780, 200, 350], [1.0, 1.0, 3.0, 2.0, 350 / 150]),
                ([1, 3, 450, 100, 150], [1, 3, 260, 200, 350], [1.0, 1.0, 260 / 450, 2.0, 350 / 150]),
                ([5, 14, 1000, 300, 40], [5, 14, 500, 140, 280], [1.0, 1.0, 0.5, 140 / 300, 7.0]),
                ([4, 3, 180, 1340], [4, 3, 60, 804], [1.0, 1.0, 0.33333334, 0.6]),
                ([4, 3, 500, 180, 1340], [4, 3, 750, 60, 804], [1.0, 1.0, 1.5, 0.33333334, 0.6])])
    def test_onnx_resize_using_scales_without_roi(self, input_shape, output_shape, scales):
        np_scales = np.array(scales)
        graph = build_graph(nodes_attrs=graph_node_attrs_scales,
                            edges=[('input', 'input_data'),
                                   ('scales', 'scales_data'),
                                   ('input_data', 'onnx_resize11', {'in': 0}),
                                   ('scales_data', 'onnx_resize11', {'in': 2}),
                                   ('onnx_resize11', 'onnx_resize11_data'),
                                   ('onnx_resize11_data', 'op_output'),
                            ],
                            update_attributes={
                                'input_data': {'shape': int64_array(input_shape)},
                                'scales': {'shape': int64_array(np_scales.shape), 'value': np_scales},
                                'scales_data': {'shape': int64_array(np_scales.shape), 'value': np_scales},
                            })
        node = Node(graph, 'onnx_resize11')
        ONNXResize11Op.onnx_resize_infer(node)

        msg = "ONNXResize11 infer failed for case: scales={}, expected_shape={}, actual_shape={}"

        self.assertTrue(np.array_equal(graph.node['onnx_resize11_data']['shape'], int64_array(output_shape)),
                        msg.format(scales, output_shape, graph.node['onnx_resize11_data']['shape']))
