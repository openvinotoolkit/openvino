# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.front.binary_quantize_normalization import BinaryFakeQuantizeNormalization
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph

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
        graph.clean_up()

        graph_ref = build_graph(graph_nodes, graph_ref_edges)
        graph_ref.clean_up()

        (flag, resp) = compare_graphs(graph, graph_ref, 'output')
        self.assertTrue(flag, resp)
