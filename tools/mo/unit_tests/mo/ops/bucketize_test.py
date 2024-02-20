# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.ops.bucketize import Bucketize
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.graph.graph import Node
from unit_tests.utils.graph import build_graph

nodes_attributes = {'input_tensor': {'shape': None, 'value': None, 'kind': 'data'},
                    'input_buckets': {'shape': None, 'value': None, 'kind': 'data'},
                    'bucketize_node': {'op': 'Bucketize', 'kind': 'op', 'with_right_bound': False, 'output_type': np.int32},
                    'output': {'shape': None, 'value': None, 'kind': 'data'}}

# graph 1
edges1 = [('input_tensor', 'bucketize_node', {'in': 0}),
          ('input_buckets', 'bucketize_node', {'in': 1}),
          ('bucketize_node', 'output', {'out': 0})]

inputs1 = {'input_tensor': {'shape': int64_array([4]), 'value': np.array([0.2, 6.4, 3.0, 1.6])},
           'input_buckets': {'shape': int64_array([5]), 'value': np.array([0.0, 1.0, 2.5, 4.0, 10.0])}}

inputs2 = {'input_tensor': {'shape': int64_array([4]), 'value': np.array([0.2, 6.4, 3.0, 1.6])},
           'input_buckets': {'shape': int64_array([5]), 'value': np.array([])}}

inputs3 = {'input_tensor': {'shape': int64_array([10, 40]), 'value': None},
           'input_buckets': {'shape': int64_array([5]), 'value': None}}

class TestBucketize(unittest.TestCase):
    def test_infer1(self):
        graph = build_graph(nodes_attributes, edges1, inputs1)
        bucketize_node = Node(graph, 'bucketize_node')
        Bucketize.infer(bucketize_node)

        # prepare reference results
        ref_output_value = np.array([1, 4, 3, 2], dtype=np.int32)

        # get the result
        res_output_value = graph.node['output']['value']

        self.assertTrue(np.array_equal(ref_output_value, res_output_value),
                        'values do not match expected: {} and given: {}'.format(ref_output_value, res_output_value))

    def test_infer2(self):
        graph = build_graph(nodes_attributes, edges1, inputs2)
        bucketize_node = Node(graph, 'bucketize_node')
        Bucketize.infer(bucketize_node)

        # prepare reference results
        ref_output_value = np.array([0, 0, 0, 0], dtype=np.int32)

        # get the result
        res_output_value = graph.node['output']['value']

        self.assertTrue(np.array_equal(ref_output_value, res_output_value),
                        'values do not match expected: {} and given: {}'.format(ref_output_value, res_output_value))

    def test_partial_infer1(self):
        graph = build_graph(nodes_attributes, edges1, inputs3)
        bucketize_node = Node(graph, 'bucketize_node')
        Bucketize.infer(bucketize_node)

        # prepare reference results
        ref_output_shape = np.array([10, 40], dtype=np.int32)

        # get the result
        res_output_shape = graph.node['output']['shape']

        self.assertTrue(np.array_equal(ref_output_shape, res_output_shape),
                        'values do not match expected: {} and given: {}'.format(ref_output_shape, res_output_shape))
