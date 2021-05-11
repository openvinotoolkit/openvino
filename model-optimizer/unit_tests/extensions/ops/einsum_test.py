# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
from generator import generator, generate

from extensions.ops.einsum import Einsum
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Graph
from mo.graph.graph import Node
from unit_tests.utils.graph import build_graph, regular_op_with_shaped_data, result, connect


def create_einsum_graph(input_shapes: list, equation: str) -> Graph:
    num_inputs = len(input_shapes)
    assert num_inputs > 0, "Einsum node must have at least one input"
    nodes = {}
    edges = []
    for input_ind in range(num_inputs):
        input_name = 'input' + str(input_ind)
        parameter_op = regular_op_with_shaped_data(input_name, input_shapes[input_ind],
                                                   {'op': 'Parameter', 'type': 'Parameter'})
        nodes.update(parameter_op)
        edges += connect(input_name, str(input_ind) + ":einsum_node")
    einsum_op = regular_op_with_shaped_data('einsum_node', None,
                                            {'op': 'Einsum', 'type': 'Einsum', 'equation': equation})
    nodes.update(einsum_op)
    result_op = result('output')
    nodes.update(result_op)
    edges += connect('einsum_node', 'output')

    graph = build_graph(nodes, edges, nodes_with_edges_only=True)
    return graph


@generator
class TestEinsum(unittest.TestCase):
    @generate(*[
        # dot product
        ([int64_array([10]), int64_array([10])], "i,i->", int64_array([])),
        # matrix multiplication
        ([int64_array([2, 3]), int64_array([3, 4])], "ab,bc->ac", int64_array([2, 4])),
        # trace per batch
        ([int64_array([2, 3, 3])], "kii->k", int64_array([2])),
        # diagonal extraction
        ([int64_array([6, 5, 5])], "kii->ki", int64_array([6, 5])),
        # transpose
        ([int64_array([1, 2, 3])], "ijk->kij", int64_array([3, 1, 2])),
        # multiple matrix multiplication
        ([int64_array([2, 5]), int64_array([5, 3, 6]), int64_array([5, 3])], "ab,bcd,bc->ca", int64_array([3, 2])),
        # ellipsis for one operand
        ([int64_array([5, 3, 4])], "a...->...", int64_array([3, 4])),
        # ellipsis for multiple operands
        ([int64_array([3, 5]), int64_array([1])], "a...,...->a...", int64_array([3, 5])),
        # ellipsis with broadcasting
        ([int64_array([9, 1, 4, 3]), int64_array([3, 11, 7, 1])], "a...b,b...->a...", int64_array([9, 11, 7, 4])),
        # mixed case letters in equation
        ([int64_array([1, 3, 5])], "AbC", int64_array([1, 5, 3])),
        # mixed case letters and equation in implicit mode
        ([int64_array([3, 11, 1, 5]), int64_array([1, 3, 1, 7])], "a...b,B...", int64_array([3, 11, 7, 1, 3, 5])),
    ])
    def test_einsum(self, input_shapes, equation, ref_output_shape):
        graph = create_einsum_graph(input_shapes, equation)
        einsum_node = Node(graph, 'einsum_node')
        Einsum.infer(einsum_node)

        # get the result
        res_output_shape = graph.node['einsum_node_d']['shape']

        self.assertTrue(np.array_equal(ref_output_shape, res_output_shape),
                        'shape does not match expected: {} and given: {}'.format(ref_output_shape, res_output_shape))

    @generate(*[
        # incorrect subscript numbers or inputs
        ([int64_array([3, 11]), int64_array([11, 4])], "ab,bc,cd->ac", None),
        # invalid labels
        ([int64_array([3, 11]), int64_array([11, 4])], "a$,Bc->ac", None),
        # incompatible shapes
        ([int64_array([3, 11]), int64_array([12, 4])], "ab,bc->ac", None),
        # not broadcastable shapes
        ([int64_array([11, 1, 4, 3]), int64_array([3, 11, 7, 5])], "a...b,b...->a...", None),
        # missed ellipsis
        ([int64_array([11, 1, 4, 3]), int64_array([3, 11, 7, 4])], "a...b,b...->a", None),
    ])
    def test_invalid_cases(self, input_shapes, equation, ref_output_shape):
        graph = create_einsum_graph(input_shapes, equation)
        einsum_node = Node(graph, 'einsum_node')
        self.assertRaises(AssertionError, Einsum.infer, einsum_node)
