# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.ops.unique import Unique
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.graph.graph import Node
from unit_tests.utils.graph import build_graph

# graph 1 with two outputs: uniques and indices
nodes_attributes = {'input': {'shape': None, 'value': None, 'kind': 'data'},
                    'unique_node': {'op': 'Unique', 'kind': 'op'},
                    'output_uniques': {'shape': None, 'value': None, 'kind': 'data'},
                    'output_indices': {'shape': None, 'value': None, 'kind': 'data'},
                    }
edges1 = [('input', 'unique_node', {'in': 0}),
          ('unique_node', 'output_uniques', {'out': 0}),
          ('unique_node', 'output_indices', {'out': 1})]
inputs1 = {'input': {'shape': int64_array([20]), 'value': None},
           'unique_node': {
               'sorted': 'false',
               'return_inverse': 'true',
               'return_counts': 'false'
               }
           }

# graph 2 with three outputs: uniques, indices and counts
nodes_attributes2 = {'input': {'shape': None, 'value': None, 'kind': 'data'},
                    'unique_node': {'op': 'Unique', 'kind': 'op'},
                    'output_uniques': {'shape': None, 'value': None, 'kind': 'data'},
                    'output_indices': {'shape': None, 'value': None, 'kind': 'data'},
                    'output_counts': {'shape': None, 'value': None, 'kind': 'data'}
                    }
edges2 = [('input', 'unique_node', {'in': 0}),
          ('unique_node', 'output_uniques', {'out': 0}),
          ('unique_node', 'output_indices', {'out': 1}),
          ('unique_node', 'output_counts', {'out': 2})]
inputs2 = {'input': {'shape': int64_array([20]), 'value': None},
           'unique_node': {
               'sorted': 'false',
               'return_inverse': 'true',
               'return_counts': 'true'
               }
           }


class TestUnique(unittest.TestCase):
    # case 1: a graph with two outputs: uniques and indices
    def test_partial_infer1(self):
        graph = build_graph(nodes_attributes, edges1, inputs1)

        unique_node = Node(graph, 'unique_node')
        Unique.infer(unique_node)

        # prepare reference results
        ref_output_uniques_shape = int64_array([20])
        ref_output_indices_shape = int64_array([20])

        # get resulted shapes
        res_output_uniques_shape = graph.node['output_uniques']['shape']
        res_output_indices_shape = graph.node['output_indices']['shape']

        self.assertTrue(np.array_equal(ref_output_uniques_shape, res_output_uniques_shape),
                        'shapes do not match expected: {} and given: {}'.format(ref_output_uniques_shape, res_output_uniques_shape))

        self.assertTrue(np.array_equal(ref_output_indices_shape, res_output_indices_shape),
                        'shapes do not match expected: {} and given: {}'.format(ref_output_indices_shape, res_output_indices_shape))

    # case 2: a graph with three outputs: uniques, indices and counts
    def test_partial_infer2(self):
        graph = build_graph(nodes_attributes2, edges2, inputs2)

        unique_node = Node(graph, 'unique_node')
        Unique.infer(unique_node)

        # prepare reference results
        ref_output_uniques_shape = int64_array([20])
        ref_output_indices_shape = int64_array([20])
        ref_output_counts_shape = int64_array([20])

        # get resulted shapes
        res_output_uniques_shape = graph.node['output_uniques']['shape']
        res_output_indices_shape = graph.node['output_indices']['shape']
        res_output_counts_shape = graph.node['output_counts']['shape']

        self.assertTrue(np.array_equal(ref_output_uniques_shape, res_output_uniques_shape),
                        'shapes do not match expected: {} and given: {}'.format(ref_output_uniques_shape, res_output_uniques_shape))

        self.assertTrue(np.array_equal(ref_output_indices_shape, res_output_indices_shape),
                        'shapes do not match expected: {} and given: {}'.format(ref_output_indices_shape, res_output_indices_shape))

        self.assertTrue(np.array_equal(ref_output_counts_shape, res_output_counts_shape),
                        'shapes do not match expected: {} and given: {}'.format(ref_output_counts_shape, res_output_counts_shape))

    # case 3: a graph with just unique output
    def test_partial_infer_just_unique(self):
        edges = [('input', 'unique_node', {'in': 0}),
                 ('unique_node', 'output_uniques', {'out': 0})]
        graph = build_graph(nodes_attributes, edges, inputs1)

        unique_node = Node(graph, 'unique_node')
        Unique.infer(unique_node)

        # prepare reference results
        ref_output_uniques_shape = int64_array([20])

        # get resulted shapes
        res_output_uniques_shape = graph.node['output_uniques']['shape']

        self.assertTrue(np.array_equal(ref_output_uniques_shape, res_output_uniques_shape),
                        'shapes do not match expected: {} and given: {}'.format(ref_output_uniques_shape, res_output_uniques_shape))

    # case 4: an invalid graph with 2D input
    def test_incorrect_input_shape(self):
        inputs = {'input': {'shape': int64_array([20, 2]), 'value': None}}

        graph = build_graph(nodes_attributes, edges1, inputs)

        unique_node = Node(graph, 'unique_node')
        self.assertRaises(AssertionError, Unique.infer, unique_node)

    # case 5: an invalid graph with return_counts = false and three outputs
    def test_more_output_ports(self):
        nodes_attributes1 = {'input': {'shape': None, 'value': None, 'kind': 'data'},
                             'unique_node': {'op': 'Unique', 'kind': 'op'},
                             'output_uniques': {'shape': None, 'value': None, 'kind': 'data'},
                             'output_indices': {'shape': None, 'value': None, 'kind': 'data'},
                             'output3': {'shape': None, 'value': None, 'kind': 'data'},
                             }
        edges = [('input', 'unique_node', {'in': 0}),
                 ('unique_node', 'output_uniques', {'out': 0}),
                 ('unique_node', 'output_indices', {'out': 1}),
                 ('unique_node', 'output3', {'out': 2})]
        graph = build_graph(nodes_attributes1, edges, inputs1)

        unique_node = Node(graph, 'unique_node')
        self.assertRaises(AssertionError, Unique.infer, unique_node)

    # case 6: an invalid graph without unique output
    def test_no_uniques_output(self):
        edges = [('input', 'unique_node', {'in': 0}),
                 ('unique_node', 'output_indices', {'out': 1})]
        graph = build_graph(nodes_attributes, edges, inputs1)

        unique_node = Node(graph, 'unique_node')
        self.assertRaises(AssertionError, Unique.infer, unique_node)

    # case 7: infer for constant input
    # graph with a constant input, three outputs, sorted = 'false'
    def test_constant_input(self):
        nodes_attributes_ = {'input': {'shape': None, 'value': None, 'kind': 'data'},
                            'unique_node': {'op': 'Unique', 'kind': 'op'},
                            'output_uniques': {'shape': None, 'value': None, 'kind': 'data'},
                            'output_indices': {'shape': None, 'value': None, 'kind': 'data'},
                            'output_counts': {'shape': None, 'value': None, 'kind': 'data'}
                            }
        edges_ = [('input', 'unique_node', {'in': 0}),
                  ('unique_node', 'output_uniques', {'out': 0}),
                  ('unique_node', 'output_indices', {'out': 1}),
                  ('unique_node', 'output_counts', {'out': 2})]
        inputs_ = {'input': {'shape': int64_array([10]),
                             'value': np.array([8.0, 1.0, 2.0, 1.0, 8.0, 5.0, 1.0, 5.0, 0.0, 0.0], dtype=float)},
                   'unique_node': {
                       'sorted': 'false',
                       'return_inverse': 'true',
                       'return_counts': 'true'
                       }
                   }
        graph = build_graph(nodes_attributes_, edges_, inputs_)
        unique_node = Node(graph, 'unique_node')
        Unique.infer(unique_node)

        # prepare reference results
        ref_output_uniques_shape = int64_array([5])
        ref_output_uniques_value = np.array([8.0, 1.0, 2.0, 5.0, 0.0], dtype=float)
        ref_output_indices_shape = int64_array([10])
        ref_output_indices_value = np.array([0.0, 1.0, 2.0, 1.0, 0.0, 3.0, 1.0, 3.0, 4.0, 4.0], dtype=float)
        ref_output_counts_shape = int64_array([5])
        ref_output_counts_value = np.array([2.0, 3.0, 1.0, 2.0, 2.0], dtype=float)

        # get resulted shapes
        res_output_uniques_shape = graph.node['output_uniques']['shape']
        res_output_uniques_value = graph.node['output_uniques']['value']
        res_output_indices_shape = graph.node['output_indices']['shape']
        res_output_indices_value = graph.node['output_indices']['value']
        res_output_counts_shape = graph.node['output_counts']['shape']
        res_output_counts_value = graph.node['output_counts']['value']

        # verify the results
        self.assertTrue(np.array_equal(ref_output_uniques_shape, res_output_uniques_shape),
                        'shapes do not match expected: {} and given: {}'.format(ref_output_uniques_shape, res_output_uniques_shape))
        self.assertTrue(np.array_equal(ref_output_uniques_value, res_output_uniques_value),
                        'values do not match expected: {} and given: {}'.format(ref_output_uniques_value, res_output_uniques_value))
        self.assertTrue(np.array_equal(ref_output_indices_shape, res_output_indices_shape),
                        'shapes do not match expected: {} and given: {}'.format(ref_output_indices_shape, res_output_indices_shape))
        self.assertTrue(np.array_equal(ref_output_indices_value, res_output_indices_value),
                        'values do not match expected: {} and given: {}'.format(ref_output_indices_value, res_output_indices_value))
        self.assertTrue(np.array_equal(ref_output_counts_shape, res_output_counts_shape),
                        'shapes do not match expected: {} and given: {}'.format(ref_output_counts_shape, res_output_counts_shape))
        self.assertTrue(np.array_equal(ref_output_counts_value, res_output_counts_value),
                        'values do not match expected: {} and given: {}'.format(ref_output_counts_value, res_output_counts_value))

    # case 8: infer for constant input
    # graph with a constant input, three outputs, sorted = 'true'
    def test_constant_input(self):
        nodes_attributes_ = {'input': {'shape': None, 'value': None, 'kind': 'data'},
                            'unique_node': {'op': 'Unique', 'kind': 'op'},
                            'output_uniques': {'shape': None, 'value': None, 'kind': 'data'},
                            'output_indices': {'shape': None, 'value': None, 'kind': 'data'},
                            'output_counts': {'shape': None, 'value': None, 'kind': 'data'}
                            }
        edges_ = [('input', 'unique_node', {'in': 0}),
                  ('unique_node', 'output_uniques', {'out': 0}),
                  ('unique_node', 'output_indices', {'out': 1}),
                  ('unique_node', 'output_counts', {'out': 2})]
        inputs_ = {'input': {'shape': int64_array([10]),
                             'value': np.array([8.0, 1.0, 2.0, 1.0, 8.0, 5.0, 1.0, 5.0, 0.0, 0.0], dtype=float)},
                   'unique_node': {
                       'sorted': 'true',
                       'return_inverse': 'true',
                       'return_counts': 'true'
                       }
                   }
        graph = build_graph(nodes_attributes_, edges_, inputs_)
        unique_node = Node(graph, 'unique_node')
        Unique.infer(unique_node)

        # prepare reference results
        ref_output_uniques_shape = int64_array([5])
        ref_output_uniques_value = np.array([0.0, 1.0, 2.0, 5.0, 8.0], dtype=float)
        ref_output_indices_shape = int64_array([10])
        ref_output_indices_value = np.array([4.0, 1.0, 2.0, 1.0, 4.0, 3.0, 1.0, 3.0, 0.0, 0.0], dtype=float)
        ref_output_counts_shape = int64_array([5])
        ref_output_counts_value = np.array([2.0, 3.0, 1.0, 2.0, 2.0], dtype=float)

        # get resulted shapes
        res_output_uniques_shape = graph.node['output_uniques']['shape']
        res_output_uniques_value = graph.node['output_uniques']['value']
        res_output_indices_shape = graph.node['output_indices']['shape']
        res_output_indices_value = graph.node['output_indices']['value']
        res_output_counts_shape = graph.node['output_counts']['shape']
        res_output_counts_value = graph.node['output_counts']['value']

        # verify the results
        self.assertTrue(np.array_equal(ref_output_uniques_shape, res_output_uniques_shape),
                        'shapes do not match expected: {} and given: {}'.format(ref_output_uniques_shape, res_output_uniques_shape))
        self.assertTrue(np.array_equal(ref_output_uniques_value, res_output_uniques_value),
                        'values do not match expected: {} and given: {}'.format(ref_output_uniques_value, res_output_uniques_value))
        self.assertTrue(np.array_equal(ref_output_indices_shape, res_output_indices_shape),
                        'shapes do not match expected: {} and given: {}'.format(ref_output_indices_shape, res_output_indices_shape))
        self.assertTrue(np.array_equal(ref_output_indices_value, res_output_indices_value),
                        'values do not match expected: {} and given: {}'.format(ref_output_indices_value, res_output_indices_value))
        self.assertTrue(np.array_equal(ref_output_counts_shape, res_output_counts_shape),
                        'shapes do not match expected: {} and given: {}'.format(ref_output_counts_shape, res_output_counts_shape))
        self.assertTrue(np.array_equal(ref_output_counts_value, res_output_counts_value),
                        'values do not match expected: {} and given: {}'.format(ref_output_counts_value, res_output_counts_value))
