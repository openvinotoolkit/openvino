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

from mo.front.common.partial_infer.eltwise import eltwise_infer
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph

nodes_attributes = {'node_1': {'value': 2, 'kind': 'data'},
                    'node_2': {'value': 3, 'kind': 'data'},
                    'eltw_1': {'kind': 'op'},
                    'node_3': {'value': None, 'kind': 'data'},
                    'op_output': { 'kind': 'op', 'op': 'Result'},
                    }


class TestEltwiseInfer(unittest.TestCase):
    def test_eltwise_infer_max(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'eltw_1'),
                             ('node_2', 'eltw_1'),
                             ('eltw_1', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': np.array([1, 3, 256, 256])},
                             'node_2': {'shape': np.array([1, 3, 256, 256])},
                             'eltw_1': {}
                             })

        graph.graph['layout'] = 'NCHW'

        eltwise_node = Node(graph, 'eltw_1')

        eltwise_infer(eltwise_node, lambda a, b: np.maximum(a, b))
        exp_shape = np.array([1, 3, 256, 256])
        exp_value = 3
        res_shape = graph.node['node_3']['shape']
        res_value = eltwise_node.out_node().value
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

        self.assertEqual(exp_value, res_value)

    def test_eltwise_infer_sum(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'eltw_1'),
                             ('node_2', 'eltw_1'),
                             ('eltw_1', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': np.array([1, 3, 256, 256])},
                             'node_2': {'shape': np.array([1, 3, 256, 256])}
                             })
        graph.graph['layout'] = 'NCHW'
        eltwise_node = Node(graph, 'eltw_1')

        eltwise_infer(eltwise_node, lambda a, b: a + b)
        exp_shape = np.array([1, 3, 256, 256])
        exp_value = 5
        res_shape = graph.node['node_3']['shape']
        res_value = eltwise_node.out_node().value
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

        self.assertEqual(exp_value, res_value)

    def test_eltwise_infer_mul(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'eltw_1'),
                             ('node_2', 'eltw_1'),
                             ('eltw_1', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': np.array([1, 3, 256, 256])},
                             'node_2': {'shape': np.array([1, 3, 256, 256])}
                             })
        graph.graph['layout'] = 'NCHW'
        eltwise_node = Node(graph, 'eltw_1')

        eltwise_infer(eltwise_node, lambda a, b: a * b)
        exp_shape = np.array([1, 3, 256, 256])
        exp_value = 6
        res_shape = graph.node['node_3']['shape']
        res_value = eltwise_node.out_node().value
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

        self.assertEqual(exp_value, res_value)

    def test_eltwise_infer_none_val(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'eltw_1'),
                             ('node_2', 'eltw_1'),
                             ('eltw_1', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': np.array([1, 3, 256, 256]), 'value': None},
                             'node_2': {'shape': np.array([1, 3, 256, 256])}
                             })
        graph.graph['layout'] = 'NCHW'
        eltwise_node = Node(graph, 'eltw_1')

        eltwise_infer(eltwise_node, lambda a, b: a * b)
        exp_shape = np.array([1, 3, 256, 256])
        res_shape = graph.node['node_3']['shape']
        res_value = eltwise_node.out_node().value
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

        self.assertIsNone(res_value)

    def test_eltwise_infer_none_min_max(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'eltw_1'),
                             ('node_2', 'eltw_1'),
                             ('eltw_1', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': np.array([1, 3, 257, 256])},
                             'node_2': {'shape': np.array([1, 3, 256, 257])}
                             })
        graph.graph['layout'] = 'NCHW'
        eltwise_node = Node(graph, 'eltw_1')

        eltwise_infer(eltwise_node)
        exp_shape = np.array([1, 3, -1, -1])
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])
