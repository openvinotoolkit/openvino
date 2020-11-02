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

from extensions.middle.MXTileReplacer import MXTileReplacer
from mo.front.common.partial_infer.utils import int64_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph

nodes_attributes = {
    'placeholder': {'kind': 'op', 'op': 'Parameter'},
    'placeholder_data': {'kind': 'data'},
    'tile': {'kind': 'op', 'op': 'Tile'},
    'tile_data': {'kind': 'data', 'shape': int64_array([1, 1, 1, 1])},
    'result': {'kind': 'op', 'op': 'Result'},

    'unsqueeze_1': {'kind': 'op', 'op': 'Unsqueeze'},
    'unsqueeze_1_data': {'kind': 'data'},
    'unsqueeze_1_const': {'kind': 'op', 'op': 'Const'},
    'unsqueeze_1_const_data': {'kind': 'data'},
}


class MXTileReplacerTest(unittest.TestCase):

    def test_insert_one_unsqueeze(self):
        graph = build_graph(
            nodes_attributes,
            [
                ('placeholder', 'placeholder_data'),
                ('placeholder_data', 'tile'),
                ('tile', 'tile_data'),
                ('tile_data', 'result')
            ],
            {
                'placeholder_data': {'shape': int64_array([1, 1, 1])}
            },
            nodes_with_edges_only=True
        )

        ref_graph = build_graph(
            nodes_attributes,
            [
                ('placeholder', 'placeholder_data'),
                ('placeholder_data', 'unsqueeze_1', {'in': 0}),
                ('unsqueeze_1_const', 'unsqueeze_1_const_data'),
                ('unsqueeze_1_const_data', 'unsqueeze_1', {'in': 1}),
                ('unsqueeze_1', 'unsqueeze_1_data'),
                ('unsqueeze_1_data', 'tile'),
                ('tile', 'tile_data'),
                ('tile_data', 'result')
            ],
            {
                'placeholder_data': {'shape': int64_array([1, 1, 1])},
                'unsqueeze_1_const_data': {'value': int64_array([0])}
            },
            nodes_with_edges_only=True
        )

        MXTileReplacer().find_and_replace_pattern(graph)
        graph.clean_up()

        (flag, resp) = compare_graphs(graph, ref_graph, 'placeholder', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_insert_two_unsqueezes(self):
        graph = build_graph(
            nodes_attributes,
            [
                ('placeholder', 'placeholder_data'),
                ('placeholder_data', 'tile'),
                ('tile', 'tile_data'),
                ('tile_data', 'result')
            ],
            {
                'placeholder_data': {'shape': int64_array([1, 1])}
            },
            nodes_with_edges_only=True
        )

        ref_graph = build_graph(
            nodes_attributes,
            [
                ('placeholder', 'placeholder_data'),
                ('placeholder_data', 'unsqueeze_1', {'in': 0}),
                ('unsqueeze_1_const', 'unsqueeze_1_const_data'),
                ('unsqueeze_1_const_data', 'unsqueeze_1', {'in': 1}),
                ('unsqueeze_1', 'unsqueeze_1_data'),
                ('unsqueeze_1_data', 'tile'),
                ('tile', 'tile_data'),
                ('tile_data', 'result')
            ],
            {
                'placeholder_data': {'shape': int64_array([1, 1])},
                'unsqueeze_1_const_data': {'value': int64_array([0, 1])}
            },
            nodes_with_edges_only=True
        )

        MXTileReplacer().find_and_replace_pattern(graph)
        graph.clean_up()

        (flag, resp) = compare_graphs(graph, ref_graph, 'placeholder', check_op_attrs=True)
        self.assertTrue(flag, resp)
