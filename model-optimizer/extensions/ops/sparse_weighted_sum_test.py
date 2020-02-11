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

from extensions.ops.sparse_weighted_sum import ExperimentalSparseWeightedSum
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph


nodes_attributes = {'input_indices': {'shape': None, 'value': None, 'kind': 'data'},
                    'input_values': {'shape': None, 'value': None, 'kind': 'data'},
                    'input_dense_shape': {'shape': None, 'value': None, 'kind': 'data'},
                    'input_params_table': {'shape': None, 'value': None, 'kind': 'data'},
                    'input_default_value': {'shape': None, 'value': None, 'kind': 'data'},
                    'input_weights': {'shape': None, 'value': None, 'kind': 'data'},
                    'sparse_weighted_sum_node': {'op': 'ExperimentalSparseWeightedSum', 'kind': 'op'},
                    'output': {'shape': None, 'value': None, 'kind': 'data'}}

# graph 1
edges1 = [('input_indices', 'sparse_weighted_sum_node', {'in': 0}),
          ('input_values', 'sparse_weighted_sum_node', {'in': 1}),
          ('input_dense_shape', 'sparse_weighted_sum_node', {'in': 2}),
          ('input_params_table', 'sparse_weighted_sum_node', {'in': 3}),
          ('input_default_value', 'sparse_weighted_sum_node', {'in': 4}),
          ('input_weights', 'sparse_weighted_sum_node', {'in': 5}),
          ('sparse_weighted_sum_node', 'output', {'out': 0})]

inputs1 = {'input_indices': {'shape': int64_array([5, 2]), 'value': None},
           'input_values': {'shape': int64_array([5]), 'value': None},
           'input_dense_shape': {'shape': int64_array([2]), 'value': int64_array([4, 3])},
           'input_params_table': {'shape': int64_array([100, 4, 5]), 'value': None},
           'input_default_value': {'shape': int64_array([]), 'value': 100.0},
           'input_weights': {'shape': int64_array([5]), 'value': None}}


class TestExperimentalSparseWeightedSum(unittest.TestCase):
    def test_partial_infer1(self):
        graph = build_graph(nodes_attributes, edges1, inputs1)
        sparse_weighted_sum_node = Node(graph, 'sparse_weighted_sum_node')
        ExperimentalSparseWeightedSum.infer(sparse_weighted_sum_node)

        # prepare reference results
        ref_output_shape = np.array([4, 4, 5], dtype=np.int32)

        # get the result
        res_output_shape = graph.node['output']['shape']

        self.assertTrue(np.array_equal(ref_output_shape, res_output_shape),
                        'shapes do not match expected: {} and given: {}'.format(ref_output_shape, res_output_shape))


