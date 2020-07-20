"""
 Copyright (C) 2020 Intel Corporation

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

from extensions.ops.scatternd import ScatterNDUpdate
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph

nodes_attributes = {'input': {'shape': None, 'value': None, 'kind': 'data'},
                    'indices': {'shape': None, 'value': None, 'kind': 'data'},
                    'updates': {'shape': None, 'value': None, 'kind': 'data'},
                    'scatternd_node': {'op': 'ScatterNDUpdate', 'kind': 'op'},
                    'output': {'shape': None, 'value': None, 'kind': 'data'}}

# graph 1
edges = [('input', 'scatternd_node', {'in': 0}),
         ('indices', 'scatternd_node', {'in': 1}),
         ('updates', 'scatternd_node', {'in': 2}),
         ('scatternd_node', 'output', {'out': 0})]

# test data for partial infer
inputs1 = {'input': {'shape': int64_array([10, 40]), 'value': None},
          'indices': {'shape': int64_array([3, 2]), 'value': None},
          'updates': {'shape': int64_array([3]), 'value': None}}

inputs2 = {'input': {'shape': int64_array([20, 30]), 'value': None},
           'indices': {'shape': int64_array([2]), 'value': None},
           'updates': {'shape': int64_array([]), 'value': None}}

inputs3 = {'input': {'shape': int64_array([20, 30, 5]), 'value': None},
           'indices': {'shape': int64_array([2]), 'value': None},
           'updates': {'shape': int64_array([5]), 'value': None}}

inputs4 = {'input': {'shape': int64_array([10, 40, 50]), 'value': None},
          'indices': {'shape': int64_array([7, 3, 2]), 'value': None},
          'updates': {'shape': int64_array([7, 3, 50]), 'value': None}}

# test data for constant folding
inputs5 = {'input': {'shape': int64_array([8]), 'value': int64_array([1, 2, 3, 4, 5, 6, 7, 8])},
          'indices': {'shape': int64_array([4, 1]), 'value': int64_array([[4], [3], [1], [7]])},
          'updates': {'shape': int64_array([4]), 'value': int64_array([9, 10, 11, 12])}}
output5 = int64_array([1, 11, 3, 10, 9, 6, 7, 12])

inputs6 = {'input': {'shape': int64_array([4, 4, 4]), 'value': int64_array([[[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
                                                                            [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
                                                                            [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
                                                                            [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]]])},
          'indices': {'shape': int64_array([2, 1]), 'value': int64_array([[0], [2]])},
          'updates': {'shape': int64_array([2, 4, 4]), 'value': int64_array([[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
                                                                             [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]])}}
output6 = int64_array([[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
                       [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
                       [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
                       [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]]])

inputs7 = {'input': {'shape': int64_array([8]), 'value': int64_array([1, 2, 3, 4, 5, 6, 7, 8])},
          'indices': {'shape': int64_array([1]), 'value': int64_array([4])},
          'updates': {'shape': int64_array([]), 'value': 9}}
output7 = int64_array([1, 2, 3, 4, 9, 6, 7, 8])

class TestScatterNDUpdate(unittest.TestCase):
    def test_partial_infer1(self):
        graph = build_graph(nodes_attributes, edges, inputs1)
        scatternd_node = Node(graph, 'scatternd_node')
        ScatterNDUpdate.infer(scatternd_node)

        # prepare reference results
        ref_output_shape = np.array([10, 40], dtype=np.int32)

        # get the result
        res_output_shape = graph.node['output']['shape']

        self.assertTrue(np.array_equal(ref_output_shape, res_output_shape),
                        'values do not match expected: {} and given: {}'.format(ref_output_shape, res_output_shape))

    def test_partial_infer2(self):
        graph = build_graph(nodes_attributes, edges, inputs2)
        scatternd_node = Node(graph, 'scatternd_node')
        ScatterNDUpdate.infer(scatternd_node)

        # prepare reference results
        ref_output_shape = np.array([20, 30], dtype=np.int32)

        # get the result
        res_output_shape = graph.node['output']['shape']

        self.assertTrue(np.array_equal(ref_output_shape, res_output_shape),
                        'values do not match expected: {} and given: {}'.format(ref_output_shape, res_output_shape))

    def test_partial_infer3(self):
        graph = build_graph(nodes_attributes, edges, inputs3)
        scatternd_node = Node(graph, 'scatternd_node')
        ScatterNDUpdate.infer(scatternd_node)

        # prepare reference results
        ref_output_shape = np.array([20, 30, 5], dtype=np.int32)

        # get the result
        res_output_shape = graph.node['output']['shape']

        self.assertTrue(np.array_equal(ref_output_shape, res_output_shape),
                        'values do not match expected: {} and given: {}'.format(ref_output_shape, res_output_shape))

    def test_partial_infer4(self):
        graph = build_graph(nodes_attributes, edges, inputs4)
        scatternd_node = Node(graph, 'scatternd_node')
        ScatterNDUpdate.infer(scatternd_node)

        # prepare reference results
        ref_output_shape = np.array([10, 40, 50], dtype=np.int32)

        # get the result
        res_output_shape = graph.node['output']['shape']

        self.assertTrue(np.array_equal(ref_output_shape, res_output_shape),
                        'values do not match expected: {} and given: {}'.format(ref_output_shape, res_output_shape))

    def test_infer5(self):
        graph = build_graph(nodes_attributes, edges, inputs5)
        scatternd_node = Node(graph, 'scatternd_node')
        ScatterNDUpdate.infer(scatternd_node)

        # get the result
        res_output_value = graph.node['output']['value']

        self.assertTrue(np.array_equal(output5, res_output_value),
                        'values do not match expected: {} and given: {}'.format(output5, res_output_value))

    def test_infer6(self):
        graph = build_graph(nodes_attributes, edges, inputs6)
        scatternd_node = Node(graph, 'scatternd_node')
        ScatterNDUpdate.infer(scatternd_node)

        # get the result
        res_output_value = graph.node['output']['value']

        self.assertTrue(np.array_equal(output6, res_output_value),
                        'values do not match expected: {} and given: {}'.format(output5, res_output_value))

    def test_infer7_scalar(self):
        graph = build_graph(nodes_attributes, edges, inputs7)
        scatternd_node = Node(graph, 'scatternd_node')
        ScatterNDUpdate.infer(scatternd_node)

        # get the result
        res_output_value = graph.node['output']['value']

        self.assertTrue(np.array_equal(output7, res_output_value),
                        'values do not match expected: {} and given: {}'.format(output5, res_output_value))
