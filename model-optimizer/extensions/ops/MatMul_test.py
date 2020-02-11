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
from generator import generator, generate

from extensions.ops.MatMul import MatMul
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph_with_attrs


@generator
class TestMatMul(unittest.TestCase):
    nodes = [
        ('A', {'type': 'Parameter', 'kind': 'op'}),
        ('A_d', {'kind': 'data'}),
        ('B', {'type': 'Parameter', 'kind': 'op'}),
        ('B_d', {'kind': 'data', 'dim_attrs': []}),
        ('mat_mul', {'type': 'MatMul', 'kind': 'op'}),
        ('mat_mul_d', {'kind': 'data', 'value': None, 'shape': None}),
        ('op_output', {'kind': 'op', 'op': 'Result'}),
    ]
    edges = [
        ('A', 'A_d'),
        ('B', 'B_d'),
        ('A_d', 'mat_mul', {'in': 0}),
        ('B_d', 'mat_mul', {'in': 1}),
        ('mat_mul', 'mat_mul_d'),
        ('mat_mul_d', 'op_output'),
    ]

    @generate(*[
        ([1024], [1024, 1000], [1, 1000], False, False),
        ([1, 1024], [1024, 1000], [1, 1000], False, False),
        ([1, 1024], [1000, 1024], [1, 1000], False, True),
        ([1024], [1024, 1000], [1, 1000], False, False),
        ([10, 1024], [1024, 1000], [10, 1000], False, False),
        ([5, 10, 1024], [1024, 1000], [5, 10, 1000], False, False),
        ([5, 10, 1024], [5, 1024, 1000], [5, 10, 1000], False, False),
        ([5, 10, 1024], [1, 1024, 1000], [5, 10, 1000], False, False),
        ([5, 10, 1024], [1, 1000, 1024], [5, 10, 1000], False, True),

    ])
    def test_positive_matmul_infer(self, A_shape, B_shape, C_shape, transpose_a, transpose_b):
        graph = build_graph_with_attrs(nodes_with_attrs=self.nodes, edges_with_attrs=self.edges,
                                       update_nodes_attributes=[
                                           ('A_d', {'shape': int64_array(A_shape)}),
                                           ('B_d', {'shape': int64_array(B_shape)}),
                                           ('mat_mul', {'transpose_a': transpose_a, 'transpose_b': transpose_b}),
                                       ])
        node = Node(graph, 'mat_mul')
        MatMul.infer(node)

        msg = "MatMul infer failed for case: A_shape={}, B_shape={}, transpose_a={}, transpose_b={}" \
              "expexted_shape={}, actual_shape={}"

        self.assertTrue(np.array_equal(graph.node['mat_mul_d']['shape'], int64_array(C_shape)),
                        msg.format(A_shape, B_shape, transpose_a, transpose_b, C_shape,
                                   graph.node['mat_mul_d']['shape']))

    @generate(*[
        (None, [1024, 1000]),
        (1, [1024, 1000]),
        ([], [1024, 1000]),
        ([1024, 1000], [1024, 1000]),
        ([5, 10, 1024], [3, 1024, 1000]),
    ])
    def test_negative_matmul_infer(self, A_shape, B_shape):
        graph = build_graph_with_attrs(nodes_with_attrs=self.nodes, edges_with_attrs=self.edges,
                                       update_nodes_attributes=[
                                           ('A_d', {'shape': np.array(A_shape)}),
                                           ('B_d', {'shape': int64_array(B_shape)}),
                                       ])

        node = Node(graph, 'mat_mul')
        self.assertRaises(AssertionError, MatMul.infer, node)
