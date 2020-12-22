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

from extensions.middle.DropoutWithRandomUniformReplacer import DropoutWithRandomUniformReplacer
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph

nodes_attributes = {
    'shape': {'kind': 'op', 'op': 'ShapeOf'},
    'shape_data': {'kind': 'data'},
    'random_uniform': {'kind': 'op', 'op': 'RandomUniform'},
    'random_uniform_data': {'kind': 'data'},
    'mul': {'kind': 'op', 'op': 'Mul'},
    'mul_const': {'kind': 'op', 'op': 'Const'},
    'mul_const_data': {'kind': 'data', 'value': np.array([1], dtype=np.int32)},
    'mul_data': {'kind': 'data'},
    'add': {'kind': 'op', 'op': 'Add'},
    'add_data': {'kind': 'data'},
    'add_const': {'kind': 'op', 'op': 'Const'},
    'add_const_data': {'kind': 'data', 'value': np.array([0], dtype=np.int32)},
    'add2': {'kind': 'op', 'op': 'Add'},
    'add2_data': {'kind': 'data'},
    'add2_const': {'kind': 'op', 'op': 'Const'},
    'add2_const_data': {'kind': 'data', 'value': np.array([1], dtype=np.int32)},
    'floor': {'kind': 'op', 'op': 'Floor', 'name': 'dropout/Floor'},
    'floor_data': {'kind': 'data'},
    'output': {'kind': 'op', 'op': 'Result'},

    # nodes used in the transformation
    'broadcast': {'kind': 'op', 'op': 'Broadcast'},
    'broadcast_const': {'kind': 'op', 'op': 'Const'},
    'broadcast_const_data': {'kind': 'data', 'value': np.array([1], dtype=np.int32)},
    'broadcast_data': {'kind': 'data'},
}


class DropoutWithRandomUniformReplacerReplacer(unittest.TestCase):
    def test_1(self):
        graph = build_graph(nodes_attributes,
                            edges=[
                                ('shape', 'shape_data'),
                                ('shape_data', 'random_uniform'),
                                ('random_uniform', 'random_uniform_data'),
                                ('random_uniform_data', 'mul', {'in': 0}),
                                ('mul_const', 'mul_const_data'),
                                ('mul_const_data', 'mul', {'in': 1}),
                                ('mul', 'mul_data'),
                                ('mul_data', 'add', {'in': 0}),
                                ('add_const', 'add_const_data'),
                                ('add_const_data', 'add', {'in': 1}),
                                ('add', 'add_data'),
                                ('add_data', 'add2', {'in': 0}),
                                ('add2_const', 'add2_const_data'),
                                ('add2_const_data', 'add2', {'in': 1}),
                                ('add2', 'add2_data'),
                                ('add2_data', 'floor'),
                                ('floor', 'floor_data'),
                                ('floor_data', 'output')
                            ],
                            nodes_with_edges_only=True)

        ref_graph = build_graph(nodes_attributes,
                                edges=[
                                    ('shape', 'shape_data'),
                                    ('shape_data', 'broadcast', {'in': 1}),
                                    ('broadcast_const', 'broadcast_const_data'),
                                    ('broadcast_const_data', 'broadcast', {'in': 0}),
                                    ('broadcast', 'broadcast_data'),
                                    ('broadcast_data', 'output'),
                                ],
                                nodes_with_edges_only=True)

        DropoutWithRandomUniformReplacer().find_and_replace_pattern(graph)

        flag, resp = compare_graphs(graph, ref_graph, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.assertTrue(graph.node[graph.get_nodes_with_attributes(op='Broadcast')[0]]['name'] == 'dropout/Floor')
