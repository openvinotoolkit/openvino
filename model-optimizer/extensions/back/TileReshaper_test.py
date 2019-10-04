"""
 Copyright (c) 2018-2019 Intel Corporation

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

from extensions.back.TileReshaper import TileReshaper
from mo.front.common.partial_infer.utils import int64_array
from mo.middle.passes.eliminate import shape_inference
from mo.ops.tile import Tile
from mo.utils.unittest.graph import build_graph, compare_graphs

# The dictionary with nodes attributes used to build various graphs. A key is the name of the node and the value is the
# dictionary with node attributes.
nodes_attributes = {
    'prev_op': {'kind': 'op', 'op': 'SomeOp'},
    'previous_data': {'shape': np.array([1, 1, 101]), 'kind': 'data'},
    'tile': {'type': 'Tile', 'kind': 'op', 'axis': 1, 'tiles': 16, 'infer': Tile.infer},
    'tile_data': {'shape': np.array([1, 16, 101]), 'kind': 'data'},
    'next_op': {'kind': 'op', 'op': 'SomeOp'},
}
edge_attributes = [
    ('prev_op', 'previous_data'),
    ('previous_data', 'tile'),
    ('tile', 'tile_data'),
    ('tile_data', 'next_op'),
]

nodes_attributes_ref = {
    'prev_op': {'kind': 'op', 'op': 'SomeOp'},
    'previous_data': {'kind': 'data', 'shape': np.array([1, 1, 101])},
    'reshape_op_before': {'type': 'Unsqueeze', 'kind': 'op'},
    'reshape_op_before_const': {'kind': 'op', 'type': 'Const', 'op': 'Const', 'value': int64_array([3]), 'shape': [1]},
    'reshape_op_before_const_data': {'kind': 'data', 'value': int64_array([3]), 'shape': [1]},
    'reshape_data_before': {'kind': 'data', 'shape': np.array([1, 1, 101, 1])},
    'tile': {'type': 'Tile', 'kind': 'op', 'infer': Tile.infer, 'axis': 1, 'tiles': 16},
    'tile_data': {'shape': np.array([1, 16, 101, 1]), 'kind': 'data'},
    'reshape_op_after': {'type': 'Squeeze', 'kind': 'op'},
    'reshape_op_after_const': {'kind': 'op', 'type': 'Const', 'op': 'Const', 'value': int64_array([3]), 'shape': [1]},
    'reshape_op_after_const_data': {'kind': 'data', 'value': int64_array([3]), 'shape': [1]},
    'reshape_data_after': {'kind': 'data', 'shape': np.array([1, 16, 101])},
    'next_op': {'kind': 'op', 'op': 'SomeOp'},
}
edge_attributes_ref = [
    ('prev_op', 'previous_data'),
    ('previous_data', 'reshape_op_before'),
    ('reshape_op_before_const', 'reshape_op_before_const_data'),
    ('reshape_op_before_const_data', 'reshape_op_before'),
    ('reshape_op_before', 'reshape_data_before'),
    ('reshape_data_before', 'tile'),
    ('tile', 'tile_data'),
    ('tile_data', 'reshape_op_after'),
    ('reshape_op_after_const', 'reshape_op_after_const_data'),
    ('reshape_op_after_const_data', 'reshape_op_after'),
    ('reshape_op_after', 'reshape_data_after'),
    ('reshape_data_after', 'next_op')
]


class TileReshaperTests(unittest.TestCase):
    def test_tile_reshaper(self):
        graph = build_graph(nodes_attributes, edge_attributes)

        graph_ref = build_graph(nodes_attributes_ref, edge_attributes_ref)

        TileReshaper().find_and_replace_pattern(graph)
        shape_inference(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'next_op', check_op_attrs=True)
        self.assertTrue(flag, resp)
