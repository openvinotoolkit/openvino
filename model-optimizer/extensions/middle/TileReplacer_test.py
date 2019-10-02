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

from extensions.middle.TileReplacer import TileMultipleAxisReplacer
from mo.utils.unittest.graph import build_graph, compare_graphs


class TileMultipleAxisReplacerTest(unittest.TestCase):
    nodes = {
        'input_op': {'kind': 'op'},
        'input_op_data': {'kind': 'data', 'shape': np.array([1, 1, 5]), 'value': None},

        'tile': {'op': 'Tile', 'kind': 'op'},
        'tile_data': {'kind': 'data', 'shape': np.array([3, 4, 5]), 'value': None},

        'next_op': {'kind': 'op', 'op': 'Result'},
        'next_op_data': {'kind': 'data', 'shape': np.array([3, 4, 5]), 'value': None},
}
    edges = [('input_op', 'input_op_data'),
             ('input_op_data', 'tile'),
             ('tile', 'tile_data'),
             ('tile_data', 'next_op'),
             ('next_op', 'next_op_data'),
             ]

    nodes_result = {
        'input_op': {'kind': 'op'},
        'input_op_data': {'kind': 'data', 'shape': np.array([1, 1, 5]), 'value': None},

        'tile_1': {'op': 'Tile', 'kind': 'op'},
        'tile_1_data': {'kind': 'data', 'shape': np.array([3, 1, 5]), 'value': None},

        'tile_2': {'op': 'Tile', 'kind': 'op'},
        'tile_2_data': {'kind': 'data', 'shape': np.array([3, 4, 5]), 'value': None},

        'next_op': {'kind': 'op', 'op': 'Result'},
        # 'next_op_data': {'kind': 'data', 'shape': np.array([3, 4, 5]), 'value': None},
    }
    edges_result = [('input_op', 'input_op_data'),
             ('input_op_data', 'tile_1'),
             ('tile_1', 'tile_1_data'),
             ('tile_1_data', 'tile_2'),
             ('tile_2', 'tile_2_data'),
             ('tile_2_data', 'next_op'),
             # ('next_op', 'next_op_data'),
             ]

    # Test for case when we don't need to add new Tiles
    def test_simple_tile(self):
        graph = build_graph(nodes_attrs=self.nodes, edges=self.edges,
                            update_attributes={'tile': {'tile_array': [3, 1, 1]},
                                               'input_op_data': {'shape': np.array([1, 4, 5])}})
        pattern = TileMultipleAxisReplacer()
        pattern.find_and_replace_pattern(graph)

        graph_ref = build_graph(nodes_attrs=self.nodes, edges=self.edges,
                            update_attributes={'tile': {'tile_array': [3, 1, 1]},
                                               'input_op_data': {'shape': np.array([1, 4, 5])}})

        (flag, resp) = compare_graphs(graph, graph_ref, 'next_op', check_op_attrs=True)
        self.assertTrue(flag, resp)

    # Test for case when we tile 2 dimensions
    def test_need_to_tile_case(self):
        graph = build_graph(nodes_attrs=self.nodes, edges=self.edges,
                            update_attributes={'tile': {'tile_array': np.array([3, 4, 1])}})
        pattern = TileMultipleAxisReplacer()
        pattern.find_and_replace_pattern(graph)

        graph_ref = build_graph(nodes_attrs=self.nodes_result, edges=self.edges_result)

        graph.clean_up()
        (flag, resp) = compare_graphs(graph, graph_ref, 'next_op', check_op_attrs=True)
        self.assertTrue(flag, resp)
