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

from extensions.ops.argmax import ArgMaxOp
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph

nodes_attributes = {'node_1': {'type': 'Identity', 'kind': 'op'},
                    'argmax': {'type': 'ArgMax', 'kind': 'op'},
                    'node_3': {'type': 'Identity', 'kind': 'op'},
                    'op_output': { 'kind': 'op', 'op': 'OpOutput'}
                    }


class TestArgMaxOp(unittest.TestCase):
    def test_caffe_argmax_axis(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'argmax'),
                             ('argmax', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': np.array([1, 3, 1025, 2049])},
                             'argmax': {
                                 'out_max_val': True,
                                 'top_k': 100,
                                 'axis': 2
                             }
                             })

        argmax_node = Node(graph, 'argmax')
        ArgMaxOp.argmax_infer(argmax_node)
        exp_shape = np.array([1, 3, 100, 2049])
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

    def test_caffe_argmax_axis_negative(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'argmax'),
                             ('argmax', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': np.array([1, 3, 1025, 2049])},
                             'argmax': {
                                 'out_max_val': True,
                                 'top_k': 100,
                                 'axis': -1
                             }
                             })

        argmax_node = Node(graph, 'argmax')
        ArgMaxOp.argmax_infer(argmax_node)
        exp_shape = np.array([1, 3, 1025, 100])
        res_shape = graph.node['node_3']['shape']
        self.assertEqual(argmax_node.axis, 3)
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

    def test_caffe_argmax_no_axis(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'argmax'),
                             ('argmax', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': np.array([1, 3, 1025, 2049])},
                             'argmax': {
                                 'out_max_val': True,
                                 'top_k': 100
                             }
                             })

        argmax_node = Node(graph, 'argmax')
        ArgMaxOp.argmax_infer(argmax_node)
        exp_shape = np.array([1, 2, 100, 1])
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

    def test_caffe_argmax_extend_shape(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'argmax'),
                             ('argmax', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': np.array([1, 3])},
                             'argmax': {
                                 'out_max_val': True,
                                 'top_k': 100
                             }
                             })

        argmax_node = Node(graph, 'argmax')
        ArgMaxOp.argmax_infer(argmax_node)
        exp_shape = np.array([1, 2, 100])
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

    def test_caffe_argmax_out_max_val_false(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'argmax'),
                             ('argmax', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': np.array([1, 3])},
                             'argmax': {
                                 'out_max_val': False,
                                 'top_k': 100
                             }
                             })

        argmax_node = Node(graph, 'argmax')
        ArgMaxOp.argmax_infer(argmax_node)
        exp_shape = np.array([1, 1, 100])
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

    def test_caffe_argmax_no_shape(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'argmax'),
                             ('argmax', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': None},
                             'argmax': {
                                 'out_max_val': False,
                                 'top_k': 100
                             }
                             })

        argmax_node = Node(graph, 'argmax')
        ArgMaxOp.argmax_infer(argmax_node)
        res_shape = graph.node['node_3']['shape']
        self.assertIsNone(res_shape)
