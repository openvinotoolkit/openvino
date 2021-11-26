# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import unittest

import numpy as np

from extensions.front.tf.ComplexAbsAfterComplex import ComplexAbsAfterComplex
from mo.front.common.partial_infer.utils import int64_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph


graph_node_attrs = {
    'placeholder_0': {'shape': int64_array([3, 100, 100, 2]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_1': {'shape': int64_array([3, 100, 100, 2]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'complex': {'kind': 'op', 'op': 'Complex'},
    'complex_abs': {'kind': 'op', 'op': 'ComplexAbs'},
    'relu': {'type': 'ReLU', 'kind': 'op', 'op': 'ReLU'},
    'output': {'type': None, 'value': None, 'kind': 'op', 'op': 'Result'},
}

graph_edges = [
    ('placeholder_0', 'complex', {'in': 0}),
    ('placeholder_1', 'complex', {'in': 1}),
    ('complex', 'complex_abs', {'in': 0}),
    ('complex_abs', 'relu'),
    ('relu', 'output'),
]


ref_graph_node_attrs = {
    'placeholder_0': {'shape': int64_array([3, 100, 100, 2]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_1': {'shape': int64_array([3, 100, 100, 2]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'pow0_const': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'value': np.float32(2.0)
    },
    'pow1_const': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'value': np.float32(2.0)
    },
    'pow0': {'type': 'Power', 'kind': 'op', 'op': 'Pow'},
    'pow1': {'type': 'Power', 'kind': 'op', 'op': 'Pow'},
    'add': {'type': 'Add', 'kind': 'op', 'op': 'Add'},
    'sqrt_const': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'value': np.float32(0.5)
    },
    'sqrt': {'type': 'Power', 'kind': 'op', 'op': 'Pow'},
    'relu': {'type': 'ReLU', 'kind': 'op', 'op': 'ReLU'},
    'output': {'type': None, 'value': None, 'kind': 'op', 'op': 'Result'},
}

ref_graph_edges = [
    ('placeholder_0', 'pow0', {'in': 0}),
    ('placeholder_1', 'pow1', {'in': 0}),
    ('pow0_const', 'pow0', {'in': 1}),
    ('pow1_const', 'pow1', {'in': 1}),
    ('pow0', 'add', {'in': 0}),
    ('pow1', 'add', {'in': 1}),
    ('add', 'sqrt', {'in': 0}),
    ('sqrt_const', 'sqrt', {'in': 1}),
    ('sqrt', 'relu'),
    ('relu', 'output'),
]


class ComplexAbsAfterComplexTest(unittest.TestCase):
    def test_replacement(self):
        graph = build_graph(nodes_attrs=graph_node_attrs, edges=graph_edges)
        graph.stage = 'front'
        ComplexAbsAfterComplex().find_and_replace_pattern(graph)
        ref_graph = build_graph(nodes_attrs=ref_graph_node_attrs, edges=ref_graph_edges)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)
