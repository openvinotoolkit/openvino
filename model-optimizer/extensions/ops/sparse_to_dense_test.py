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

from extensions.ops.sparse_to_dense import SparseToDense
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph

# graph 1
nodes_attributes = {
    'input_indices': {'kind': 'op', 'op': 'Parameter', 'shape': int64_array([5, 2])},
    'input_indices_data': {'kind': 'data', 'shape': int64_array([5, 2]), 'value': None},
    'input_dense_shape': {'kind': 'op', 'op': 'Const', 'shape': int64_array([2])},
    'input_dense_shape_data': {'kind': 'data', 'shape': int64_array([2]), 'value': int64_array([4, 3])},
    'input_values': {'kind': 'op', 'op': 'Parameter', 'shape': int64_array([5])},
    'input_values_data': {'kind': 'data', 'shape': int64_array([5]), 'value': None},
    'input_default_value': {'kind': 'op', 'op': 'Parameter', 'shape': int64_array([])},
    'input_default_value_data': {'kind': 'data', 'shape': int64_array([]), 'value': 100},
    'sparse_to_dense_node': {'type': 'SparseToDense', 'op': 'SparseToDense', 'kind': 'op'},
    'output_data': {'kind': 'data', 'shape': None, 'value': None}
}

edges = [('input_indices', 'input_indices_data'),
         ('input_dense_shape', 'input_dense_shape_data'),
         ('input_values', 'input_values_data'),
         ('input_default_value', 'input_default_value_data'),
         ('input_indices_data', 'sparse_to_dense_node', {'in': 0}),
         ('input_dense_shape_data', 'sparse_to_dense_node', {'in': 1}),
         ('input_values_data', 'sparse_to_dense_node', {'in': 2}),
         ('input_default_value_data', 'sparse_to_dense_node', {'in': 3}),
         ('sparse_to_dense_node', 'output_data', {'out': 0})]

inputs = {'input_indices_data': {'shape': int64_array([5, 2]), 'value': int64_array([[0, 0], [0, 1], [2, 1], [2, 2], [3, 0]])},
          'input_dense_shape_data': {'shape': int64_array([2]), 'value': int64_array([4, 3])},
          'input_values_data': {'shape': int64_array([5]), 'value': np.array([1.0, 2.0, 10.0, 4.0, 7.0])},
          'input_default_value_data': {'shape': int64_array([]), 'value': 100.0}}


class TestSparseToDense(unittest.TestCase):
    def test_partial_infer1(self):
        graph = build_graph(nodes_attributes, edges)
        sparse_to_dense_node = Node(graph, 'sparse_to_dense_node')
        SparseToDense.infer(sparse_to_dense_node)

        # prepare reference results
        ref_output_shape = np.array([4, 3], dtype=np.int32)

        # get the result
        res_output_shape = graph.node['output_data']['shape']

        self.assertTrue(np.array_equal(ref_output_shape, res_output_shape),
                        'shapes do not match expected: {} and given: {}'.format(ref_output_shape, res_output_shape))

    def test_infer1(self):
        graph = build_graph(nodes_attributes, edges, inputs)
        sparse_to_dense_node = Node(graph, 'sparse_to_dense_node')
        SparseToDense.infer(sparse_to_dense_node)

        # prepare reference results
        ref_output_value = np.array([[1.0, 2.0, 100.0],
                                     [100.0, 100.0, 100.0],
                                     [100.0, 10.0, 4.0],
                                     [7.0, 100.0, 100.0]])

        # get the result
        res_output_value = graph.node['output_data']['value']

        self.assertTrue(np.array_equal(ref_output_value, res_output_value),
                        'values do not match expected: {} and given: {}'.format(ref_output_value, res_output_value))
