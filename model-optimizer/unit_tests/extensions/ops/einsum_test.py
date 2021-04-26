# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from extensions.ops.einsum import Einsum
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node
from unit_tests.utils.graph import build_graph

nodes_attributes = {'input1': {'kind': 'op'},
                    'input1_data': {'shape': None, 'value': None, 'kind': 'data'},
                    'input2': {'kind': 'op'},
                    'input2_data': {'shape': None, 'value': None, 'kind': 'data'},
                    'input3': {'kind': 'op'},
                    'input3_data': {'shape': None, 'value': None, 'kind': 'data'},
                    'einsum_node': {'op': 'Einsum', 'kind': 'op', 'equation': None},
                    'output': {'shape': None, 'value': None, 'kind': 'data'}}

# invalid test case with incorrect rank for indices
inputs_inv1 = {'data_data': {'shape': int64_array([10, 40]), 'value': None},
               'indices_data': {'shape': int64_array([5, 3, 4]), 'value': None}}


class TestEinsum(unittest.TestCase):
    def test_einsum_dotproduct(self):
        edges = [('input1', 'input1_data', {'in': 0}),
                 ('input2', 'input2_data', {'in': 0}),
                 ('input1_data', 'einsum_node', {'in': 0}),
                 ('input2_data', 'einsum_node', {'in': 1}),
                 ('einsum_node', 'output', {'out': 0})]
        inputs = {'input1_data': {'shape': int64_array([10])},
                  'input2_data': {'shape': int64_array([10])},
                  'einsum_node': {'equation': "i,i->"}}
        ref_output_shape = int64_array([])

        graph = build_graph(nodes_attributes, edges, inputs, nodes_with_edges_only=True)
        einsum_node = Node(graph, 'einsum_node')
        Einsum.infer(einsum_node)

        # get the result
        res_output_shape = graph.node['output']['shape']

        self.assertTrue(np.array_equal(ref_output_shape, res_output_shape),
                        'shape does not match expected: {} and given: {}'.format(ref_output_shape, res_output_shape))

    def test_einsum_matmul(self):
        edges = [('input1', 'input1_data', {'in': 0}),
                 ('input2', 'input2_data', {'in': 0}),
                 ('input1_data', 'einsum_node', {'in': 0}),
                 ('input2_data', 'einsum_node', {'in': 1}),
                 ('einsum_node', 'output', {'out': 0})]
        inputs = {'input1_data': {'shape': int64_array([2, 3])},
                  'input2_data': {'shape': int64_array([3, 4])},
                  'einsum_node': {'equation': "ab,bc->ac"}}
        ref_output_shape = int64_array([2, 4])

        graph = build_graph(nodes_attributes, edges, inputs, nodes_with_edges_only=True)
        einsum_node = Node(graph, 'einsum_node')
        Einsum.infer(einsum_node)

        # get the result
        res_output_shape = graph.node['output']['shape']

        self.assertTrue(np.array_equal(ref_output_shape, res_output_shape),
                        'shape does not match expected: {} and given: {}'.format(ref_output_shape, res_output_shape))

    def test_einsum_trace(self):
        edges = [('input1', 'input1_data', {'in': 0}),
                 ('input1_data', 'einsum_node', {'in': 0}),
                 ('einsum_node', 'output', {'out': 0})]
        inputs = {'input1_data': {'shape': int64_array([2, 3, 3])},
                  'einsum_node': {'equation': "kii->k"}}
        ref_output_shape = int64_array([2])

        graph = build_graph(nodes_attributes, edges, inputs, nodes_with_edges_only=True)
        einsum_node = Node(graph, 'einsum_node')
        Einsum.infer(einsum_node)

        # get the result
        res_output_shape = graph.node['output']['shape']

        self.assertTrue(np.array_equal(ref_output_shape, res_output_shape),
                        'shape does not match expected: {} and given: {}'.format(ref_output_shape, res_output_shape))

    def test_einsum_diagextraction(self):
        edges = [('input1', 'input1_data', {'in': 0}),
                 ('input1_data', 'einsum_node', {'in': 0}),
                 ('einsum_node', 'output', {'out': 0})]
        inputs = {'input1_data': {'shape': int64_array([6, 5, 5])},
                  'einsum_node': {'equation': "kii->ki"}}
        ref_output_shape = int64_array([6, 5])

        graph = build_graph(nodes_attributes, edges, inputs, nodes_with_edges_only=True)
        einsum_node = Node(graph, 'einsum_node')
        Einsum.infer(einsum_node)

        # get the result
        res_output_shape = graph.node['output']['shape']

        self.assertTrue(np.array_equal(ref_output_shape, res_output_shape),
                        'shape does not match expected: {} and given: {}'.format(ref_output_shape, res_output_shape))

    def test_einsum_transpose(self):
        edges = [('input1', 'input1_data', {'in': 0}),
                 ('input1_data', 'einsum_node', {'in': 0}),
                 ('einsum_node', 'output', {'out': 0})]
        inputs = {'input1_data': {'shape': int64_array([1, 2, 3])},
                  'einsum_node': {'equation': "ijk->kij"}}
        ref_output_shape = int64_array([3, 1, 2])

        graph = build_graph(nodes_attributes, edges, inputs, nodes_with_edges_only=True)
        einsum_node = Node(graph, 'einsum_node')
        Einsum.infer(einsum_node)

        # get the result
        res_output_shape = graph.node['output']['shape']

        self.assertTrue(np.array_equal(ref_output_shape, res_output_shape),
                        'shape does not match expected: {} and given: {}'.format(ref_output_shape, res_output_shape))

    def test_einsum_multimatmul(self):
        edges = [('input1', 'input1_data', {'in': 0}),
                 ('input2', 'input2_data', {'in': 0}),
                 ('input3', 'input3_data', {'in': 0}),
                 ('input1_data', 'einsum_node', {'in': 0}),
                 ('input2_data', 'einsum_node', {'in': 1}),
                 ('input3_data', 'einsum_node', {'in': 2}),
                 ('einsum_node', 'output', {'out': 0})]
        inputs = {'input1_data': {'shape': int64_array([2, 5])},
                  'input2_data': {'shape': int64_array([5, 3, 6])},
                  'input3_data': {'shape': int64_array([5, 3])},
                  'einsum_node': {'equation': "ab,bcd,bc->ca"}}
        ref_output_shape = int64_array([3, 2])

        graph = build_graph(nodes_attributes, edges, inputs, nodes_with_edges_only=True)
        einsum_node = Node(graph, 'einsum_node')
        Einsum.infer(einsum_node)

        # get the result
        res_output_shape = graph.node['output']['shape']

        self.assertTrue(np.array_equal(ref_output_shape, res_output_shape),
                        'shape does not match expected: {} and given: {}'.format(ref_output_shape, res_output_shape))

    def test_einsum_ellipsis(self):
        edges = [('input1', 'input1_data', {'in': 0}),
                 ('input1_data', 'einsum_node', {'in': 0}),
                 ('einsum_node', 'output', {'out': 0})]
        inputs = {'input1_data': {'shape': int64_array([5, 3, 4])},
                  'einsum_node': {'equation': "a...->..."}}
        ref_output_shape = int64_array([3, 4])

        graph = build_graph(nodes_attributes, edges, inputs, nodes_with_edges_only=True)
        einsum_node = Node(graph, 'einsum_node')
        Einsum.infer(einsum_node)

        # get the result
        res_output_shape = graph.node['output']['shape']

        self.assertTrue(np.array_equal(ref_output_shape, res_output_shape),
                        'shape does not match expected: {} and given: {}'.format(ref_output_shape, res_output_shape))

    def test_einsum_ellipsis2(self):
        edges = [('input1', 'input1_data', {'in': 0}),
                 ('input2', 'input2_data', {'in': 0}),
                 ('input1_data', 'einsum_node', {'in': 0}),
                 ('input2_data', 'einsum_node', {'in': 1}),
                 ('einsum_node', 'output', {'out': 0})]
        inputs = {'input1_data': {'shape': int64_array([3, 5])},
                  'input2_data': {'shape': int64_array([1])},
                  'einsum_node': {'equation': "a...,...->a..."}}
        ref_output_shape = int64_array([3, 5])

        graph = build_graph(nodes_attributes, edges, inputs, nodes_with_edges_only=True)
        einsum_node = Node(graph, 'einsum_node')
        Einsum.infer(einsum_node)

        # get the result
        res_output_shape = graph.node['output']['shape']

        self.assertTrue(np.array_equal(ref_output_shape, res_output_shape),
                        'shape does not match expected: {} and given: {}'.format(ref_output_shape, res_output_shape))

    def test_einsum_ellipsis3(self):
        edges = [('input1', 'input1_data', {'in': 0}),
                 ('input2', 'input2_data', {'in': 0}),
                 ('input1_data', 'einsum_node', {'in': 0}),
                 ('input2_data', 'einsum_node', {'in': 1}),
                 ('einsum_node', 'output', {'out': 0})]
        inputs = {'input1_data': {'shape': int64_array([9, 1, 4, 3])},
                  'input2_data': {'shape': int64_array([3, 11, 7, 1])},
                  'einsum_node': {'equation': "a...b,b...->a..."}}
        ref_output_shape = int64_array([9, 11, 7, 4])

        graph = build_graph(nodes_attributes, edges, inputs, nodes_with_edges_only=True)
        einsum_node = Node(graph, 'einsum_node')
        Einsum.infer(einsum_node)

        # get the result
        res_output_shape = graph.node['output']['shape']

        self.assertTrue(np.array_equal(ref_output_shape, res_output_shape),
                        'shape does not match expected: {} and given: {}'.format(ref_output_shape, res_output_shape))

    def test_einsum_implicitmode_mixedcaseletters(self):
        edges = [('input1', 'input1_data', {'in': 0}),
                 ('input1_data', 'einsum_node', {'in': 0}),
                 ('einsum_node', 'output', {'out': 0})]
        inputs = {'input1_data': {'shape': int64_array([1, 3, 5])},
                  'einsum_node': {'equation': "AbC"}}
        ref_output_shape = int64_array([1, 5, 3])

        graph = build_graph(nodes_attributes, edges, inputs, nodes_with_edges_only=True)
        einsum_node = Node(graph, 'einsum_node')
        Einsum.infer(einsum_node)

        # get the result
        res_output_shape = graph.node['output']['shape']

        self.assertTrue(np.array_equal(ref_output_shape, res_output_shape),
                        'shape does not match expected: {} and given: {}'.format(ref_output_shape, res_output_shape))

    def test_einsum_implicitmode_mixedcaseletters2(self):
        edges = [('input1', 'input1_data', {'in': 0}),
                 ('input1_data', 'einsum_node', {'in': 0}),
                 ('input2', 'input2_data', {'in': 0}),
                 ('input2_data', 'einsum_node', {'in': 1}),
                 ('einsum_node', 'output', {'out': 0})]
        inputs = {'input1_data': {'shape': int64_array([3, 11, 1, 5])},
                  'input2_data': {'shape': int64_array([1, 3, 1, 7])},
                  'einsum_node': {'equation': "a...b,B..."}}
        ref_output_shape = int64_array([3, 11, 7, 1, 3, 5])

        graph = build_graph(nodes_attributes, edges, inputs, nodes_with_edges_only=True)
        einsum_node = Node(graph, 'einsum_node')
        Einsum.infer(einsum_node)

        # get the result
        res_output_shape = graph.node['output']['shape']

        self.assertTrue(np.array_equal(ref_output_shape, res_output_shape),
                        'shape does not match expected: {} and given: {}'.format(ref_output_shape, res_output_shape))

    def test_incorrect_subscript_number(self):
        edges = [('input1', 'input1_data', {'in': 0}),
                 ('input1_data', 'einsum_node', {'in': 0}),
                 ('input2', 'input2_data', {'in': 0}),
                 ('input2_data', 'einsum_node', {'in': 1}),
                 ('einsum_node', 'output', {'out': 0})]
        inputs = {'input1_data': {'shape': int64_array([3, 11])},
                  'input2_data': {'shape': int64_array([11, 4])},
                  'einsum_node': {'equation': "ab,bc,cd->ac"}}

        graph = build_graph(nodes_attributes, edges, inputs, nodes_with_edges_only=True)
        einsum_node = Node(graph, 'einsum_node')
        self.assertRaises(AssertionError, Einsum.infer, einsum_node)

    def test_invalid_labels(self):
        edges = [('input1', 'input1_data', {'in': 0}),
                 ('input1_data', 'einsum_node', {'in': 0}),
                 ('input2', 'input2_data', {'in': 0}),
                 ('input2_data', 'einsum_node', {'in': 1}),
                 ('einsum_node', 'output', {'out': 0})]
        inputs = {'input1_data': {'shape': int64_array([3, 11])},
                  'input2_data': {'shape': int64_array([11, 4])},
                  'einsum_node': {'equation': "a$,Bc->ac"}}

        graph = build_graph(nodes_attributes, edges, inputs, nodes_with_edges_only=True)
        einsum_node = Node(graph, 'einsum_node')
        self.assertRaises(AssertionError, Einsum.infer, einsum_node)

    def test_incompatible_shapes(self):
        edges = [('input1', 'input1_data', {'in': 0}),
                 ('input1_data', 'einsum_node', {'in': 0}),
                 ('input2', 'input2_data', {'in': 0}),
                 ('input2_data', 'einsum_node', {'in': 1}),
                 ('einsum_node', 'output', {'out': 0})]
        inputs = {'input1_data': {'shape': int64_array([3, 11])},
                  'input2_data': {'shape': int64_array([12, 4])},
                  'einsum_node': {'equation': "ab,bc->ac"}}

        graph = build_graph(nodes_attributes, edges, inputs, nodes_with_edges_only=True)
        einsum_node = Node(graph, 'einsum_node')
        self.assertRaises(AssertionError, Einsum.infer, einsum_node)

    def test_notbroadcastable_shapes(self):
        edges = [('input1', 'input1_data', {'in': 0}),
                 ('input1_data', 'einsum_node', {'in': 0}),
                 ('input2', 'input2_data', {'in': 0}),
                 ('input2_data', 'einsum_node', {'in': 1}),
                 ('einsum_node', 'output', {'out': 0})]
        inputs = {'input1_data': {'shape': int64_array([11, 1, 4, 3])},
                  'input2_data': {'shape': int64_array([3, 11, 7, 5])},
                  'einsum_node': {'equation': "a...b,b...->a..."}}

        graph = build_graph(nodes_attributes, edges, inputs, nodes_with_edges_only=True)
        einsum_node = Node(graph, 'einsum_node')
        self.assertRaises(AssertionError, Einsum.infer, einsum_node)

    def test_missed_ellipsis(self):
        edges = [('input1', 'input1_data', {'in': 0}),
                 ('input1_data', 'einsum_node', {'in': 0}),
                 ('input2', 'input2_data', {'in': 0}),
                 ('input2_data', 'einsum_node', {'in': 1}),
                 ('einsum_node', 'output', {'out': 0})]
        inputs = {'input1_data': {'shape': int64_array([11, 1, 4, 3])},
                  'input2_data': {'shape': int64_array([3, 11, 7, 4])},
                  'einsum_node': {'equation': "a...b,b...->a"}}

        graph = build_graph(nodes_attributes, edges, inputs, nodes_with_edges_only=True)
        einsum_node = Node(graph, 'einsum_node')
        self.assertRaises(AssertionError, Einsum.infer, einsum_node)
