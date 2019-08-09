"""
 Copyright (c) 2019 Intel Corporation

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

from extensions.front.binary_quantize_normalization import BinaryFakeQuantizeNormalization
from mo.middle.passes.eliminate import graph_clean_up
from mo.utils.unittest.graph import build_graph, compare_graphs

graph_nodes = {
    '0': {'name': 'input', 'kind': 'op', 'op': 'Parameter'},
    '1': {'name': 'mi_i', 'kind': 'op', 'op': 'Const'},
    '2': {'name': 'ma_i', 'kind': 'op', 'op': 'Const'},
    '3': {'name': 'mi_o', 'kind': 'op', 'op': 'Const'},
    '4': {'name': 'mi_o', 'kind': 'op', 'op': 'Const'},

    'add': {'kind': 'op', 'op': 'Add'},
    'const': {'kind': 'op', 'op': 'Const', 'value': np.array(0.5)},
    'mul': {'kind': 'op', 'op': 'Mul'},

    'quantize': {'name': 'quantize', 'levels': 2, 'kind': 'op', 'op': 'FakeQuantize'},

    'output': {'name': 'output1', 'kind': 'op', 'op': 'Result', 'type': 'Result'},
}

graph_edges = [
    ('0', 'quantize', {'in': 0}),
    ('1', 'quantize', {'in': 1}),
    ('2', 'quantize', {'in': 2}),
    ('3', 'quantize', {'in': 3}),
    ('4', 'quantize', {'in': 4}),
    ('quantize', 'output'),
]

graph_ref_edges = [
    ('0', 'quantize', {'in': 0}),
    ('1', 'add'),
    ('2', 'add'),
    ('add', 'mul'),
    ('const', 'mul'),
    ('mul', 'quantize', {'in': 1, 'out': 0}),
    ('mul', 'quantize', {'in': 2, 'out': 0}),
    ('3', 'quantize', {'in': 3}),
    ('4', 'quantize', {'in': 4}),
    ('quantize', 'output'),
]


class TestBinaryQuantizeNormalization(unittest.TestCase):
    def test_binary_quantize_normalizer(self):
        graph = build_graph(graph_nodes, graph_edges, nodes_with_edges_only=True)
        graph.stage = 'front'
        BinaryFakeQuantizeNormalization().find_and_replace_pattern(graph)
        graph_clean_up(graph)

        graph_ref = build_graph(graph_nodes, graph_ref_edges)
        graph_clean_up(graph_ref)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output')
        self.assertTrue(flag, resp)
