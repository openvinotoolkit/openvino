"""
 Copyright (C) 2018-2020 Intel Corporation

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

from extensions.back.TileNormalizer import TileMultipleAxisReplacer, Tile3DReshaper
from mo.front.common.partial_infer.utils import int64_array
from mo.ops.tile import Tile
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph


class TileMultipleAxisReplacerTest(unittest.TestCase):
    nodes = {
        'input_op': {'kind': 'op'},
        'input_op_data': {'kind': 'data', 'shape': np.array([1, 1, 5]), 'value': None},

        'const': {'kind': 'op'},
        'const_data': {'kind': 'data', 'value': None},

        'tile': {'op': 'Tile', 'kind': 'op'},
        'tile_data': {'kind': 'data', 'shape': np.array([3, 4, 5]), 'value': None},

        'next_op': {'kind': 'op', 'op': 'Result'},
        'next_op_data': {'kind': 'data', 'shape': np.array([3, 4, 5]), 'value': None},
    }
    edges = [('input_op', 'input_op_data'),
             ('input_op_data', 'tile', {'in': 0}),
             ('tile', 'tile_data'),
             ('tile_data', 'next_op'),
             ('next_op', 'next_op_data'),

             ('const', 'const_data'),
             ('const_data', 'tile', {'in': 1})
             ]

    nodes_result = {
        'input_op': {'kind': 'op'},
        'input_op_data': {'kind': 'data', 'shape': np.array([1, 1, 5]), 'value': None},

        'const': {'kind': 'op'},
        'const_data': {'kind': 'data'},

        'tile_1': {'op': 'Tile', 'kind': 'op'},
        'tile_1_data': {'kind': 'data', 'shape': np.array([3, 1, 5]), 'value': None},

        'const_1': {'kind': 'op'},
        'const_1_data': {'kind': 'data'},

        'tile_2': {'op': 'Tile', 'kind': 'op'},
        'tile_2_data': {'kind': 'data', 'shape': np.array([3, 4, 5]), 'value': None},

        'next_op': {'kind': 'op', 'op': 'Result'},
    }
    edges_result = [
        ('input_op', 'input_op_data'),
        ('input_op_data', 'tile_1'),
        ('tile_1', 'tile_1_data'),
        ('tile_1_data', 'tile_2'),
        ('tile_2', 'tile_2_data'),
        ('tile_2_data', 'next_op'),

        ('const', 'const_data'),
        ('const_data', 'tile_1'),

        ('const_1', 'const_1_data'),
        ('const_1_data', 'tile_2'),
    ]

    # Test for case when we don't need to add new Tiles
    def test_simple_tile(self):
        graph = build_graph(nodes_attrs=self.nodes, edges=self.edges,
                            update_attributes={'const_data': {'value': np.array([3, 1, 1])},
                                               'input_op_data': {'shape': np.array([1, 4, 5])}})
        pattern = TileMultipleAxisReplacer()
        pattern.find_and_replace_pattern(graph)

        graph_ref = build_graph(nodes_attrs=self.nodes, edges=self.edges,
                                update_attributes={'const_data': {'value': np.array([3, 1, 1])},
                                                   'input_op_data': {'shape': np.array([1, 4, 5])}})

        (flag, resp) = compare_graphs(graph, graph_ref, 'next_op', check_op_attrs=True)
        self.assertTrue(flag, resp)

    # Test for case when we tile 2 dimensions
    def test_need_to_tile_case(self):
        graph = build_graph(nodes_attrs=self.nodes, edges=self.edges,
                            update_attributes={'const_data': {'value': np.array([3, 4, 1])}})
        pattern = TileMultipleAxisReplacer()
        pattern.find_and_replace_pattern(graph)

        graph_ref = build_graph(nodes_attrs=self.nodes_result, edges=self.edges_result)

        graph.clean_up()
        (flag, resp) = compare_graphs(graph, graph_ref, 'next_op', check_op_attrs=True)
        self.assertTrue(flag, resp)


nodes_attributes = {
    'prev_op': {'kind': 'op', 'op': 'SomeOp'},
    'previous_data': {'shape': np.array([1, 1, 101]), 'value': None, 'kind': 'data'},

    'const': {'kind': 'op'},
    'const_data': {'kind': 'data', 'value': np.array([1, 16, 1]), 'shape': np.array([3])},

    'tile': {'type': 'Tile', 'kind': 'op', 'infer': Tile.infer},
    'tile_data': {'shape': np.array([1, 16, 101]), 'value': None, 'kind': 'data'},
    'next_op': {'kind': 'op', 'op': 'Result'},
}
edge_attributes = [
    ('prev_op', 'previous_data'),
    ('previous_data', 'tile', {'in': 0}),
    ('tile', 'tile_data'),
    ('tile_data', 'next_op'),

    ('const', 'const_data'),
    ('const_data', 'tile', {'in': 1}),
]

nodes_attributes_ref = {
    'prev_op': {'kind': 'op', 'op': 'SomeOp'},
    'previous_data': {'kind': 'data', 'shape': np.array([1, 1, 101])},
    'reshape_op_before': {'type': 'Unsqueeze', 'kind': 'op'},
    'reshape_op_before_const': {'kind': 'op', 'type': 'Const', 'op': 'Const', 'value': int64_array([3]), 'shape': [1]},
    'reshape_op_before_const_data': {'kind': 'data', 'value': int64_array([3]), 'shape': [1]},
    'reshape_data_before': {'kind': 'data', 'shape': np.array([1, 1, 101, 1])},

    'const': {'kind': 'op'},
    'const_data': {'kind': 'data', 'value': np.array([1, 16, 1, 1]), 'shape': np.array([4])},

    'tile': {'type': 'Tile', 'kind': 'op', 'infer': Tile.infer},
    'tile_data': {'shape': np.array([1, 16, 101, 1]), 'kind': 'data'},
    'reshape_op_after': {'type': 'Squeeze', 'kind': 'op'},
    'reshape_op_after_const': {'kind': 'op', 'type': 'Const', 'op': 'Const', 'value': int64_array([3]), 'shape': [1]},
    'reshape_op_after_const_data': {'kind': 'data', 'value': int64_array([3]), 'shape': [1]},
    'reshape_data_after': {'kind': 'data', 'shape': np.array([1, 16, 101])},
    'next_op': {'kind': 'op', 'op': 'Result'},
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
    ('reshape_data_after', 'next_op'),

    ('const', 'const_data'),
    ('const_data', 'tile', {'in': 1}),
]


class TileReshaperTests(unittest.TestCase):
    def test_tile_reshaper(self):
        graph = build_graph(nodes_attributes, edge_attributes)

        graph_ref = build_graph(nodes_attributes_ref, edge_attributes_ref)

        Tile3DReshaper().find_and_replace_pattern(graph)
        graph.clean_up()

        (flag, resp) = compare_graphs(graph, graph_ref, 'next_op', check_op_attrs=True)
        self.assertTrue(flag, resp)
