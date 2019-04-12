"""
 Copyright (c) 2018-2019 Intel Corporation

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

from extensions.ops.priorbox import PriorBoxOp
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph

nodes_attributes = {'node_1': {'type': 'Identity', 'value': None, 'kind': 'data'},
                    'pb': {'type': 'PriorBox', 'value': None, 'kind': 'op'},
                    'node_3': {'type': 'Identity', 'value': None, 'kind': 'data'},
                    'op_output': { 'kind': 'op', 'op': 'OpOutput'}
                    }


class TestPriorBoxPartialInfer(unittest.TestCase):
    def test_caffe_priorbox_infer(self):
        graph = build_graph(nodes_attributes,
                            [
                                ('node_1', 'pb'),
                                ('pb', 'node_3'),
                                ('node_3', 'op_output')
                            ],
                            {
                                'node_3': {'shape': None},
                                'node_1': {'shape': np.array([1, 384, 19, 19])},
                                'pb': {
                                    'aspect_ratio': np.array([1]),
                                    'flip': 0,
                                    'min_size': np.array([1]),
                                    'max_size': np.array([1])
                                }
                            })
        graph.graph['layout'] = 'NCHW'
        pb_node = Node(graph, 'pb')
        PriorBoxOp.priorbox_infer(pb_node)
        exp_shape = np.array([1, 2, 4 * 19 * 19 * 2])
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

    def test_caffe_priorbox_flip_infer(self):
        graph = build_graph(nodes_attributes,
                            [
                                ('node_1', 'pb'),
                                ('pb', 'node_3'),
                                ('node_3', 'op_output')
                            ],
                            {
                                'node_3': {'shape': None},
                                'node_1': {'shape': np.array([1, 384, 19, 19])},
                                'pb': {
                                    'aspect_ratio': np.array([1, 2, 0.5]),
                                    'flip': 1,
                                    'min_size': np.array([1]),
                                    'max_size': np.array([1])
                                }
                            })
        graph.graph['layout'] = 'NCHW'
        pb_node = Node(graph, 'pb')
        PriorBoxOp.priorbox_infer(pb_node)
        exp_shape = np.array([1, 2, 4 * 19 * 19 * 4])
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

    def test_tf_priorbox_infer(self):
        graph = build_graph(nodes_attributes,
                            [
                                ('node_1', 'pb'),
                                ('pb', 'node_3'),
                                ('node_3', 'op_output')
                            ],
                            {
                                'node_3': {'shape': None},
                                'node_1': {'shape': np.array([1, 19, 19, 384])},
                                'pb': {
                                    'aspect_ratio': np.array([1]),
                                    'flip': 0,
                                    'min_size': np.array([1]),
                                    'max_size': np.array([1])
                                }
                            })
        graph.graph['layout'] = 'NHWC'
        pb_node = Node(graph, 'pb')
        PriorBoxOp.priorbox_infer(pb_node)
        exp_shape = np.array([1, 2, 4 * 19 * 19 * 2])
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

    def test_tf_priorbox_flip_infer(self):
        graph = build_graph(nodes_attributes,
                            [
                                ('node_1', 'pb'),
                                ('pb', 'node_3'),
                                ('node_3', 'op_output')
                            ],
                            {
                                'node_3': {'shape': None},
                                'node_1': {'shape': np.array([1, 19, 19, 384])},
                                'pb': {
                                    'aspect_ratio': np.array([1, 2, 0.5]),
                                    'flip': 1,
                                    'min_size': np.array([1]),
                                    'max_size': np.array([1])
                                }
                            })
        graph.graph['layout'] = 'NHWC'
        pb_node = Node(graph, 'pb')
        PriorBoxOp.priorbox_infer(pb_node)
        exp_shape = np.array([1, 2, 4 * 19 * 19 * 4])
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])
