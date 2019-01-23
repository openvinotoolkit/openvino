"""
 Copyright (c) 2018 Intel Corporation

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

from mo.front.common.partial_infer.range import tf_range_infer
from mo.graph.graph import Node
from mo.utils.unittest.extractors import FakeParam
from mo.utils.unittest.graph import build_graph

nodes_attributes = {'start': {'kind': 'data'},
                    'limit': {'kind': 'data'},
                    'delta': {'kind': 'data'},
                    'range': {'kind': 'op'},
                    'output': {'value': None, 'shape': None, 'kind': 'data'},
                    }
edges = [('start', 'range'), ('limit', 'range'), ('delta', 'range'), ('range', 'output')]


class TestRangePartialInfer(unittest.TestCase):
    def test_int32_specific_data_type_range_infer(self):
        # import tensorflow to use TF data types
        import tensorflow as tf
        graph = build_graph(nodes_attributes, edges,
                            {'start': {'value': np.array([1])},
                             'limit': {'value': np.array([5])},
                             'delta': {'value': np.array([1])},
                             'range': {'pb': FakeParam('attr', dict(type=FakeParam('type', tf.int32)))},
                             })

        range_node = Node(graph, 'range')

        tf_range_infer(range_node)
        exp_value = np.array([1, 2, 3, 4], dtype=np.int32)
        out_value = graph.node['output']['value']

        self.assertTrue(exp_value.dtype == out_value.dtype)
        self.assertTrue(np.array_equal(exp_value.shape, out_value.shape))
        self.assertTrue(np.array_equal(exp_value, out_value))

    def test_automatic_data_type_range_infer(self):
        graph = build_graph(nodes_attributes, edges,
                            {'start': {'value': np.array([2], dtype=np.float32)},
                             'limit': {'value': np.array([5])},
                             'delta': {'value': np.array([1])},
                             'range': {'pb': FakeParam('attr', dict())},
                             })

        range_node = Node(graph, 'range')

        tf_range_infer(range_node)
        exp_value = np.array([2.0, 3.0, 4.0], dtype=np.float32)
        out_value = graph.node['output']['value']

        self.assertTrue(exp_value.dtype == out_value.dtype)
        self.assertTrue(np.array_equal(exp_value.shape, out_value.shape))
        self.assertTrue(np.array_equal(exp_value, out_value))

    def test_non_constant_start_range_infer(self):
        graph = build_graph(nodes_attributes, edges,
                            {'start': {},
                             'limit': {'value': np.array([5])},
                             'delta': {'value': np.array([1])},
                             'range': {'pb': FakeParam('attr', dict())},
                             })

        range_node = Node(graph, 'range')

        tf_range_infer(range_node)
        out_value = graph.node['output']['value']
        self.assertIsNone(out_value)

    def test_non_constant_limit_range_infer(self):
        graph = build_graph(nodes_attributes, edges,
                            {'start': {'value': np.array([1])},
                             'limit': {},
                             'delta': {'value': np.array([1])},
                             'range': {'pb': FakeParam('attr', dict())},
                             })

        range_node = Node(graph, 'range')

        tf_range_infer(range_node)
        out_value = graph.node['output']['value']
        self.assertIsNone(out_value)

    def test_non_constant_delta_range_infer(self):
        graph = build_graph(nodes_attributes, edges,
                            {'start': {'value': np.array([1])},
                             'limit': {'value': np.array([10])},
                             'delta': {},
                             'range': {'pb': FakeParam('attr', dict())},
                             })

        range_node = Node(graph, 'range')

        tf_range_infer(range_node)
        out_value = graph.node['output']['value']
        self.assertIsNone(out_value)
