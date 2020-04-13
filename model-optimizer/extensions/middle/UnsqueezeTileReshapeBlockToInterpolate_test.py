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

from extensions.middle.UnsqueezeTileReshapeBlockToInterpolate import UnsqueezeTileReshapeBlockToInterpolate
from mo.front.common.partial_infer.utils import int64_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph

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


ref_graph_node_attrs = {
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
    'strided_slice': {
        'type': 'StridedSlice',
        'kind': 'op',
        'op': 'StridedSlice',
        'begin_mask': int64_array([1]),
        'end_mask': int64_array([1]),
        'new_axis_mask': int64_array([0]),
        'shrink_axis_mask': int64_array([0]),
        'ellipsis_mask': int64_array([0]),
    },
    'strided_slice_data': {
        'kind': 'data',
        'shape': None,
        'value': None,
    },
    'begin': {
        'kind': 'op',
        'op': 'Const',
        'type': 'Const',
        'value': int64_array([1]),
        'shape': int64_array([1]),
    },
    'begin_data': {
        'kind': 'data',
        'value': None,
        'shape': None,
    },
    'end': {
        'kind': 'op',
        'op': 'Const',
        'type': 'Const',
        'value': int64_array([2]),
        'shape': int64_array([1]),
    },
    'end_data': {
        'kind': 'data',
        'value': None,
        'shape': None,
    },
    'scales': {
        'kind': 'op',
        'op': 'Const',
        'type': 'Const',
        'value': int64_array([2]),
        'shape': int64_array([1]),
    },
    'scales_data': {
        'kind': 'data',
        'value': None,
        'shape': None,
    },
    'mul': {'type': 'Mul', 'kind': 'op', 'op': 'Mul'},
    'mul_data': {
        'kind': 'data',
        'value': None,
        'shape': None,
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

ref_graph_edges = [
    ('placeholder', 'placeholder_data'),
    ('placeholder_data', 'shapeof'),
    ('shapeof', 'shapeof_data'),
    ('shapeof_data', 'strided_slice', {'in': 0}),
    ('strided_slice', 'strided_slice_data'),
    ('begin', 'begin_data'),
    ('begin_data', 'strided_slice', {'in': 1}),
    ('end', 'end_data'),
    ('end_data', 'strided_slice', {'in': 2}),
    ('scales', 'scales_data'),
    ('strided_slice_data', 'mul', {'in': 0}),
    ('scales_data', 'mul', {'in': 1}),
    ('mul', 'mul_data'),
    ('mul_data', 'interpolate', {'in': 1}),
    ('placeholder_data', 'interpolate', {'in': 0}),
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
        ref_graph = build_graph(nodes_attrs=ref_graph_node_attrs, edges=ref_graph_edges)
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
            nodes_attrs=ref_graph_node_attrs,
            edges=ref_graph_edges,
            update_attributes={
                'placeholder_data': {'shape': int64_array([1, 8, 32, 32])},
                'interpolate_data': {'shape': int64_array([1, 16, 32, 32])},
                'abs_data': {'shape': int64_array([1, 16, 32, 32])},
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
