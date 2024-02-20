# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
import numpy as np

from openvino.tools.mo.middle.UnsqueezeTileReshapeBlockToInterpolate import UnsqueezeTileReshapeBlockToInterpolate
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph

graph_node_attrs = {
    'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_data': {
        'value': None,
        'shape': int64_array([1, 8, 32, 32, 64]),
        'kind': 'data',
        'data_type': None
    },
    'unsqueeze': {'type': 'Unsqueeze', 'kind': 'op', 'op': 'Unsqueeze'},
    'dim': {
        'kind': 'op',
        'op': 'Const',
        'type': 'Const',
        'value': int64_array([2]),
        'shape': int64_array([1]),
    },
    'dim_data': {
        'kind': 'data',
        'value': int64_array([2]),
        'shape': int64_array([1]),
    },
    'unsqueeze_data': {
        'kind': 'data',
        'shape': int64_array([1, 8, 1, 32, 32, 64]),
        'value': None,
    },
    'tile': {'type': 'Tile', 'kind': 'op', 'op': 'Tile'},
    'multipliers': {
        'kind': 'op',
        'op': 'Const',
        'type': 'Const',
        'value': int64_array([1, 1, 2, 1, 1, 1]),
        'shape': int64_array([6]),
    },
    'multipliers_data': {
        'kind': 'data',
        'value': int64_array([1, 1, 2, 1, 1, 1]),
        'shape': int64_array([6]),
    },
    'tile_data': {
        'kind': 'data',
        'shape': int64_array([1, 8, 2, 32, 32, 64]),
        'value': None,
    },
    'reshape': {'type': 'Reshape', 'kind': 'op', 'op': 'Reshape'},
    'reshape_data': {
        'kind': 'data',
        'shape': int64_array([1, 16, 32, 32, 64]),
        'value': None,
    },
    'shape': {
        'kind': 'op',
        'op': 'Const',
        'type': 'Const',
        'value': int64_array([1, 16, 32, 32, 64]),
        'shape': int64_array([5]),
    },
    'shape_data': {
        'kind': 'data',
        'value': int64_array([1, 16, 32, 32, 64]),
        'shape': int64_array([5]),
    },
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'abs_data': {
        'kind': 'data',
        'shape': int64_array([1, 16, 32, 32, 64]),
        'value': None,
    },
    'output': {'kind': 'op', 'op': 'Result', 'type': 'Result'},
}

graph_edges = [
    ('placeholder', 'placeholder_data'),
    ('placeholder_data', 'unsqueeze', {'in': 0}),
    ('dim', 'dim_data'),
    ('dim_data', 'unsqueeze', {'in': 1}),
    ('unsqueeze', 'unsqueeze_data'),
    ('unsqueeze_data', 'tile', {'in': 0}),
    ('multipliers', 'multipliers_data'),
    ('multipliers_data', 'tile', {'in': 1}),
    ('tile', 'tile_data'),
    ('tile_data', 'reshape', {'in': 0}),
    ('reshape', 'reshape_data'),
    ('shape', 'shape_data'),
    ('shape_data', 'reshape', {'in': 1}),
    ('reshape_data', 'abs'),
    ('abs', 'abs_data'),
    ('abs_data', 'output'),
]


ref_graph_node_attrs_with_4_inputs_interpolate = {
    'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_data': {
        'value': None,
        'shape': int64_array([1, 8, 32, 32, 64]),
        'kind': 'data',
        'data_type': None
    },
    'shapeof': {'type': 'ShapeOf', 'kind': 'op', 'op': 'ShapeOf'},
    'shapeof_data': {
        'kind': 'data',
        'shape': None,
        'value': None,
    },
    'gather': {
        'type': 'Gather',
        'kind': 'op',
        'op': 'Gather'
    },
    'gather_data': {
        'kind': 'data',
        'shape': None,
        'value': None,
    },
    'indices': {
        'kind': 'op',
        'op': 'Const',
        'type': 'Const',
        'value': int64_array([1]),
        'shape': int64_array([1]),
    },
    'indices_data': {
        'kind': 'data',
        'value': None,
        'shape': None,
    },
    'gather_axis': {
        'kind': 'op',
        'op': 'Const',
        'type': 'Const',
        'value': np.array(0, dtype=np.int64),
        'shape': np.array(0, dtype=np.int64).shape,
    },
    'gather_axis_data': {
        'kind': 'data',
        'value': None,
        'shape': None,
    },
    'scales_m': {
        'kind': 'op',
        'op': 'Const',
        'type': 'Const',
        'value': np.array([2], dtype=np.int64),
        'shape': int64_array([1]),
    },
    'scales_m_data': {
        'kind': 'data',
        'value': np.array([2], dtype=np.float32),
        'shape': int64_array([1]),
    },
    'mul': {'type': 'Mul', 'kind': 'op', 'op': 'Mul'},
    'mul_data': {
        'kind': 'data',
        'value': None,
        'shape': None,
    },
    'scales': {
        'kind': 'op',
        'op': 'Const',
        'type': 'Const',
        'value': np.array([2], dtype=np.float32),
        'shape': int64_array([1]),
    },
    'scales_data': {
        'kind': 'data',
        'value': np.array([2], dtype=np.float32),
        'shape': int64_array([1]),
    },
    'axes': {
        'kind': 'op',
        'op': 'Const',
        'type': 'Const',
        'value': int64_array([1]),
        'shape': int64_array([1]),
    },
    'axes_data': {
        'kind': 'data',
        'value': int64_array([1]),
        'shape': int64_array([1]),
    },
    'interpolate': {'type': 'Interpolate', 'kind': 'op', 'op': 'Interpolate'},
    'interpolate_data': {
        'kind': 'data',
        'value': None,
        'shape': int64_array([1, 16, 32, 32, 64]),
    },
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'abs_data': {
        'kind': 'data',
        'shape': int64_array([1, 16, 32, 32, 64]),
        'value': None,
    },
    'output': {'kind': 'op', 'op': 'Result', 'type': 'Result'},
}


ref_graph_edges_attrs_with_4_inputs_interpolate = [
    ('placeholder', 'placeholder_data'),
    ('placeholder_data', 'shapeof'),
    ('shapeof', 'shapeof_data'),
    ('shapeof_data', 'gather', {'in': 0}),
    ('gather', 'gather_data'),
    ('indices', 'indices_data'),
    ('indices_data', 'gather', {'in': 1}),
    ('gather_axis', 'gather_axis_data'),
    ('gather_axis_data', 'gather', {'in': 2}),
    ('scales_m', 'scales_m_data'),
    ('gather_data', 'mul', {'in': 0}),
    ('scales_m_data', 'mul', {'in': 1}),
    ('mul', 'mul_data'),
    ('scales', 'scales_data'),
    ('axes', 'axes_data'),
    ('scales_data', 'interpolate', {'out': 0, 'in': 2}),
    ('mul_data', 'interpolate', {'in': 1}),
    ('placeholder_data', 'interpolate', {'in': 0}),
    ('axes_data', 'interpolate', {'in': 3}),
    ('interpolate', 'interpolate_data'),
    ('interpolate_data', 'abs'),
    ('abs', 'abs_data'),
    ('abs_data', 'output'),
]


graph_node_attrs_when_transformation_is_not_applicable = {
    'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_data': {
        'value': None,
        'shape': int64_array([1, 8, 32]),
        'kind': 'data',
        'data_type': None
    },
    'unsqueeze': {'type': 'Unsqueeze', 'kind': 'op', 'op': 'Unsqueeze'},
    'dim': {
        'kind': 'op',
        'op': 'Const',
        'type': 'Const',
        'value': int64_array([2]),
        'shape': int64_array([1]),
    },
    'dim_data': {
        'kind': 'data',
        'value': int64_array([2]),
        'shape': int64_array([1]),
    },
    'unsqueeze_data': {
        'kind': 'data',
        'shape': int64_array([1, 8, 1, 32]),
        'value': None,
    },
    'tile': {'type': 'Tile', 'kind': 'op', 'op': 'Tile'},
    'multipliers': {
        'kind': 'op',
        'op': 'Const',
        'type': 'Const',
        'value': int64_array([1, 1, 2, 1]),
        'shape': int64_array([4]),
    },
    'multipliers_data': {
        'kind': 'data',
        'value': int64_array([1, 1, 2, 1]),
        'shape': int64_array([4]),
    },
    'tile_data': {
        'kind': 'data',
        'shape': int64_array([1, 8, 2, 32]),
        'value': None,
    },
    'reshape': {'type': 'Reshape', 'kind': 'op', 'op': 'Reshape'},
    'reshape_data': {
        'kind': 'data',
        'shape': int64_array([1, 16, 32]),
        'value': None,
    },
    'shape': {
        'kind': 'op',
        'op': 'Const',
        'type': 'Const',
        'value': int64_array([1, 16, 32]),
        'shape': int64_array([3]),
    },
    'shape_data': {
        'kind': 'data',
        'value': int64_array([1, 16, 32]),
        'shape': int64_array([3]),
    },
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'abs_data': {
        'kind': 'data',
        'shape': int64_array([1, 16, 32]),
        'value': None,
    },
    'output': {'kind': 'op', 'op': 'Result', 'type': 'Result'},
}

graph_edges_when_transformation_is_not_applicable = graph_edges


class UnsqueezeTileReshapeBlockToInterpolateTest(unittest.TestCase):
    def test_5d(self):
        graph = build_graph(nodes_attrs=graph_node_attrs, edges=graph_edges)
        ref_graph = build_graph(nodes_attrs=ref_graph_node_attrs_with_4_inputs_interpolate,
                                edges=ref_graph_edges_attrs_with_4_inputs_interpolate)
        UnsqueezeTileReshapeBlockToInterpolate().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output')
        self.assertTrue(flag, resp)

    def test_4d(self):
        graph = build_graph(
            nodes_attrs=graph_node_attrs,
            edges=graph_edges,
            update_attributes={
                'placeholder_data': {'shape': int64_array([1, 8, 32, 32])},
                'unsqueeze_data': {'shape': int64_array([1, 8, 1, 32, 32])},
                'multipliers': {'value': int64_array([1, 1, 2, 1, 1]), 'shape': int64_array([5])},
                'multipliers_data': {'value': int64_array([1, 1, 2, 1, 1]), 'shape': int64_array([5])},
                'tile_data': {'shape': int64_array([1, 8, 2, 32, 32])},
                'reshape_data': {'shape': int64_array([1, 16, 32, 32]), 'value': None},
                'shape': {'value': int64_array([1, 16, 32, 32]), 'shape': int64_array([4])},
                'shape_data': {'value': int64_array([1, 16, 32, 32]), 'shape': int64_array([4])},
                'abs_data': {'shape': int64_array([1, 16, 32, 32])},
            }
        )
        ref_graph = build_graph(
            nodes_attrs=ref_graph_node_attrs_with_4_inputs_interpolate,
            edges=ref_graph_edges_attrs_with_4_inputs_interpolate,
            update_attributes={
                'placeholder_data': {'shape': int64_array([1, 8, 32, 32])},
                'interpolate_data': {'shape': int64_array([1, 16, 32, 32])},
                'abs_data': {'shape': int64_array([1, 16, 32, 32])},
                'axes': {'shape': int64_array([1]), 'value': int64_array([1])},
            }
        )
        UnsqueezeTileReshapeBlockToInterpolate().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output')
        self.assertTrue(flag, resp)

    def test_3d(self):
        graph = build_graph(
            nodes_attrs=graph_node_attrs_when_transformation_is_not_applicable,
            edges=graph_edges_when_transformation_is_not_applicable
        )
        ref_graph = build_graph(
            nodes_attrs=graph_node_attrs_when_transformation_is_not_applicable,
            edges=graph_edges_when_transformation_is_not_applicable
        )
        UnsqueezeTileReshapeBlockToInterpolate().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output')
        self.assertTrue(flag, resp)

    def test_2d(self):
        graph = build_graph(
            nodes_attrs=graph_node_attrs_when_transformation_is_not_applicable,
            edges=graph_edges_when_transformation_is_not_applicable,
            update_attributes={
                'placeholder_data': {'shape': int64_array([5, 8])},
                'dim': {'value': int64_array([1])},
                'dim_data': {'value': int64_array([1])},
                'unsqueeze_data': {'shape': int64_array([5, 1, 8])},
                'multipliers': {'value': int64_array([1, 10, 1])},
                'multipliers_data': {'value': int64_array([1, 10, 1]), 'shape': int64_array([3])},
                'tile_data': {'shape': int64_array([5, 10, 8])},
                'reshape_data': {'shape': int64_array([50, 8])},
                'shape': {'value': int64_array([50, 8]), 'shape': int64_array([2])},
                'shape_data': {'value': int64_array([50, 8]), 'shape': int64_array([2])},
                'abs_data': {'shape': int64_array([50, 8])},
            }
        )
        ref_graph = build_graph(
            nodes_attrs=graph_node_attrs_when_transformation_is_not_applicable,
            edges=graph_edges_when_transformation_is_not_applicable,
            update_attributes={
                'placeholder_data': {'shape': int64_array([5, 8])},
                'dim': {'value': int64_array([1])},
                'dim_data': {'value': int64_array([1])},
                'unsqueeze_data': {'shape': int64_array([5, 1, 8])},
                'multipliers': {'value': int64_array([1, 10, 1])},
                'multipliers_data': {'value': int64_array([1, 10, 1]), 'shape': int64_array([3])},
                'tile_data': {'shape': int64_array([5, 10, 8])},
                'reshape_data': {'shape': int64_array([50, 8])},
                'shape': {'value': int64_array([50, 8]), 'shape': int64_array([2])},
                'shape_data': {'value': int64_array([50, 8]), 'shape': int64_array([2])},
                'abs_data': {'shape': int64_array([50, 8])},
            }
        )
        UnsqueezeTileReshapeBlockToInterpolate().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output')
        self.assertTrue(flag, resp)
