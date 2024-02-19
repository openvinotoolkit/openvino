# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.ops.sparse_fill_empty_rows import SparseFillEmptyRows
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.graph.graph import Node
from unit_tests.utils.graph import build_graph

nodes_attributes = {'input_indices': {'shape': None, 'value': None, 'kind': 'data'},
                    'input_values': {'shape': None, 'value': None, 'kind': 'data'},
                    'dense_shape': {'shape': None, 'value': None, 'kind': 'data'},
                    'default_value': {'shape': None, 'value': None, 'kind': 'data'},
                    'sparse_fill_empty_rows_node': {'op': 'SparseFillEmptyRows', 'kind': 'op'},
                    'output_indices': {'shape': None, 'value': None, 'kind': 'data'},
                    'output_values': {'shape': None, 'value': None, 'kind': 'data'},
                    'empty_row_indicator': {'shape': None, 'value': None, 'kind': 'data'},
                    'result_indices': {'kind': 'op', 'op': 'Result'},
                    'result_values': {'kind': 'op', 'op': 'Result'},
                    'result_empty_row_indicator': {'kind': 'op', 'op': 'Result'},
                    }

# graph 1
edges1 = [('input_indices', 'sparse_fill_empty_rows_node', {'in': 0}),
          ('input_values', 'sparse_fill_empty_rows_node', {'in': 1}),
          ('dense_shape', 'sparse_fill_empty_rows_node', {'in': 2}),
          ('default_value', 'sparse_fill_empty_rows_node', {'in': 3}),
          ('sparse_fill_empty_rows_node', 'output_indices', {'out': 0}),
          ('sparse_fill_empty_rows_node', 'output_values', {'out': 1}),
          ('sparse_fill_empty_rows_node', 'empty_row_indicator', {'out': 2}),
          ('output_indices', 'result_indices', {'out': 0}),
          ('output_values', 'result_values', {'out': 0}),
          ('empty_row_indicator', 'result_empty_row_indicator', {'out': 0}),
          ]

inputs1 = {'input_indices': {'shape': int64_array([20, 2]), 'value': None},
           'input_values': {'shape': int64_array([20]), 'value': None},
           'dense_shape': {'shape': int64_array([2]), 'value': np.array([4, 5])},
           'default_value': {'shape': int64_array([]), 'value': None}}


class TestSparseFillEmptyRows(unittest.TestCase):
    def test_partial_infer(self):
        graph = build_graph(nodes_attributes, edges1, inputs1)

        sparse_fill_empty_rows_node = Node(graph, 'sparse_fill_empty_rows_node')
        SparseFillEmptyRows.infer(sparse_fill_empty_rows_node)

        # prepare reference results
        ref_output_indices_shape = int64_array([20, 2])
        ref_output_values_shape = int64_array([20])
        ref_empty_row_indicator_shape = int64_array([4])

        # get resulted shapes
        res_output_indices_shape = graph.node['output_indices']['shape']
        res_output_values_shape = graph.node['output_values']['shape']
        res_empty_row_indicator_shape = graph.node['empty_row_indicator']['shape']

        self.assertTrue(np.array_equal(ref_output_indices_shape, res_output_indices_shape),
                        'shapes do not match expected: {} and given: {}'.format(ref_output_indices_shape, res_output_indices_shape))

        self.assertTrue(np.array_equal(ref_output_values_shape, res_output_values_shape),
                        'shapes do not match expected: {} and given: {}'.format(ref_output_values_shape, res_output_values_shape))

        self.assertTrue(np.array_equal(ref_empty_row_indicator_shape, res_empty_row_indicator_shape),
                        'shapes do not match expected: {} and given: {}'.format(ref_empty_row_indicator_shape, res_empty_row_indicator_shape))

    def test_partial_infer_for_some_out_ports(self):
        edges = [('input_indices', 'sparse_fill_empty_rows_node', {'in': 0}),
                 ('input_values', 'sparse_fill_empty_rows_node', {'in': 1}),
                 ('dense_shape', 'sparse_fill_empty_rows_node', {'in': 2}),
                 ('default_value', 'sparse_fill_empty_rows_node', {'in': 3}),
                 ('sparse_fill_empty_rows_node', 'output_indices', {'out': 0}),
                 ('sparse_fill_empty_rows_node', 'empty_row_indicator', {'out': 2}),
                 ('output_indices', 'result_indices', {'out': 0}),
                 ('empty_row_indicator', 'result_empty_row_indicator', {'out': 0}),
                 ]
        graph = build_graph(nodes_attributes, edges, inputs1)

        sparse_fill_empty_rows_node = Node(graph, 'sparse_fill_empty_rows_node')
        SparseFillEmptyRows.infer(sparse_fill_empty_rows_node)

        # prepare reference results
        ref_output_indices_shape = int64_array([20, 2])
        ref_empty_row_indicator_shape = int64_array([4])

        # get resulted shapes
        res_output_indices_shape = graph.node['output_indices']['shape']
        res_empty_row_indicator_shape = graph.node['empty_row_indicator']['shape']

        self.assertTrue(np.array_equal(ref_output_indices_shape, res_output_indices_shape),
                        'shapes do not match expected: {} and given: {}'.format(ref_output_indices_shape, res_output_indices_shape))

        self.assertTrue(np.array_equal(ref_empty_row_indicator_shape, res_empty_row_indicator_shape),
                        'shapes do not match expected: {} and given: {}'.format(ref_empty_row_indicator_shape, res_empty_row_indicator_shape))

    def test_incorrect_shape_of_default_value(self):
        inputs = {'input_indices': {'shape': int64_array([20, 2]), 'value': None},
                   'input_values': {'shape': int64_array([20]), 'value': None},
                   'dense_shape': {'shape': int64_array([2]), 'value': np.array([4, 5])},
                   'default_value': {'shape': int64_array([3]), 'value': None}}
        graph = build_graph(nodes_attributes, edges1, inputs)
        sparse_fill_empty_rows_node = Node(graph, 'sparse_fill_empty_rows_node')
        self.assertRaises(AssertionError, SparseFillEmptyRows.infer, sparse_fill_empty_rows_node)

    def test_no_value_of_dense_shape(self):
        inputs = {'input_indices': {'shape': int64_array([20, 2]), 'value': None},
                   'input_values': {'shape': int64_array([20]), 'value': None},
                   'dense_shape': {'shape': int64_array([2]), 'value': None},
                   'default_value': {'shape': int64_array([]), 'value': None}}
        graph = build_graph(nodes_attributes, edges1, inputs)
        sparse_fill_empty_rows_node = Node(graph, 'sparse_fill_empty_rows_node')
        self.assertRaises(AssertionError, SparseFillEmptyRows.infer, sparse_fill_empty_rows_node)

    def test_incorrect_shape_of_dense_shape(self):
        inputs = {'input_indices': {'shape': int64_array([20, 2]), 'value': None},
                   'input_values': {'shape': int64_array([20]), 'value': None},
                   'dense_shape': {'shape': int64_array([2, 2]), 'value': np.array([[4, 5],[1, 2]])},
                   'default_value': {'shape': int64_array([]), 'value': None}}
        graph = build_graph(nodes_attributes, edges1, inputs)
        sparse_fill_empty_rows_node = Node(graph, 'sparse_fill_empty_rows_node')
        self.assertRaises(AssertionError, SparseFillEmptyRows.infer, sparse_fill_empty_rows_node)
