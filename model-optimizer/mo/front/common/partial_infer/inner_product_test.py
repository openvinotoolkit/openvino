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

from mo.front.common.partial_infer.inner_product import caffe_inner_product
from mo.graph.graph import Node
from mo.utils.unittest.extractors import FakeValue
from mo.utils.unittest.graph import build_graph

nodes_attributes = {'node_1': {'value': None, 'kind': 'data'},
                    'inner': {'type': 'FullyConnected', 'value': None, 'kind': 'op'},
                    'node_2': {'value': FakeValue(None), 'kind': 'data'},
                    'node_3': {'value': None, 'kind': 'data'}
                    }


class TestInnerPartialInfer(unittest.TestCase):
    def test_inner_product_infer(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'inner'),
                             ('node_2', 'inner'),
                             ('inner', 'node_3')],
                            {'node_3': {'is_output': True, 'shape': None},
                             'node_1': {'shape': np.array([1, 3, 256, 256])},
                             'node_2': {'shape': np.array([1, 3, 256, 256]),
                                        'dim_attrs': ['spatial_dims', 'channel_dims', 'batch_dims', 'axis']},
                             'inner': {'out-size': 4}
                             })

        inner_node = Node(graph, 'inner')

        caffe_inner_product(inner_node)
        exp_shape = np.array([1, 4])
        exp_shape_in_node = np.array([4, 3 * 256 * 256])
        res_shape = graph.node['node_3']['shape']
        res_shape_in_node = inner_node.in_node(1).shape
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

        for i in range(0, len(exp_shape_in_node)):
            self.assertEqual(exp_shape_in_node[i], res_shape_in_node[i])

    def test_inner_product_infer_no_shape(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'inner'),
                             ('node_2', 'inner'),
                             ('inner', 'node_3')],
                            {'node_3': {'is_output': True, 'shape': None},
                             'node_1': {'shape': None},
                             'node_2': {'shape': np.array([1, 3, 256, 256])},
                             'inner': {'out-size': 4}
                             })

        inner_node = Node(graph, 'inner')

        caffe_inner_product(inner_node)
        res_shape = graph.node['node_3']['shape']
        self.assertIsNone(res_shape)
