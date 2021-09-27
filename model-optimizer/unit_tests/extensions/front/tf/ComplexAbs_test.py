# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import unittest

import numpy as np

from extensions.front.tf.ComplexAbs import ComplexAbs
from mo.front.common.partial_infer.utils import int64_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph


graph_node_attrs = {
    'placeholder': {'shape': int64_array([3, 100, 100, 2]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'complex_abs': {'kind': 'op', 'op': 'ComplexAbs'},
    'relu': {'type': 'ReLU', 'kind': 'op', 'op': 'ReLU'},
    'output': {'type': None, 'value': None, 'kind': 'op', 'op': 'Result'},
}

graph_edges = [
    ('placeholder', 'complex_abs'),
    ('complex_abs', 'relu'),
    ('relu', 'output'),
]


ref_graph_node_attrs = {
    'placeholder': {'shape': int64_array([3, 100, 100, 2]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'pow2_const': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'value': np.float32(2.0)
    },
    'pow2': {'type': 'Power', 'kind': 'op', 'op': 'Pow'},
    'sum_axis': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'value': int64_array(-1)
    },
    'sum': {'type': 'ReduceSum', 'kind': 'op', 'op': 'ReduceSum'},
    'sqrt_const': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'value': np.float32(0.5)
    },
    'sqrt': {'type': 'Power', 'kind': 'op', 'op': 'Pow'},
    'relu': {'type': 'ReLU', 'kind': 'op', 'op': 'ReLU'},
    'output': {'type': None, 'value': None, 'kind': 'op', 'op': 'Result'},
}

ref_graph_edges = [
    ('placeholder', 'pow2', {'in': 0}),
    ('pow2_const', 'pow2', {'in': 1}),
    ('sum_axis', 'sum', {'in': 1}),
    ('pow2', 'sum', {'in': 0}),
    ('sum', 'sqrt', {'in': 0}),
    ('sqrt_const', 'sqrt', {'in': 1}),
    ('sqrt', 'relu'),
    ('relu', 'output'),
]


class ComplexAbsTest(unittest.TestCase):
    def test_replacement(self):
        graph = build_graph(nodes_attrs=graph_node_attrs, edges=graph_edges)
        graph.stage = 'front'
        ComplexAbs().find_and_replace_pattern(graph)
        ref_graph = build_graph(nodes_attrs=ref_graph_node_attrs, edges=ref_graph_edges)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)
