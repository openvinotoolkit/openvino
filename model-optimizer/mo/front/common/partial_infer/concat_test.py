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

from mo.front.common.partial_infer.concat import concat_infer
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph

nodes_attributes = {'node_1': {'kind': 'data', 'value': None},
                    'node_2': {'kind': 'data', 'value': None},
                    'concat': {'type': 'Concat', 'kind': 'op'},
                    'node_3': {'kind': 'data'},
                    'op_output': { 'kind': 'op', 'op': 'OpOutput'},
                    }


class TestConcatPartialInfer(unittest.TestCase):
    def test_tf_concat_infer(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'concat'),
                             ('node_2', 'concat'),
                             ('concat', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': np.array([1, 3, 227, 227])},
                             'node_2': {'shape': np.array([1, 3, 227, 227])},
                             'concat': {'axis': 2}
                             })

        concat_node = Node(graph, 'concat')
        concat_infer(concat_node)
        exp_shape = np.array([1, 3, 454, 227])
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

    def test_tf_concat_infer_negative_axis(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'concat'),
                             ('node_2', 'concat'),
                             ('concat', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': np.array([1, 3, 227, 227])},
                             'node_2': {'shape': np.array([1, 3, 227, 227])},
                             'concat': {'axis': -1}
                             })

        concat_node = Node(graph, 'concat')
        concat_infer(concat_node)
        exp_shape = np.array([1, 3, 227, 454])
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

    def test_tf_concat_infer_not_match(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'concat'),
                             ('node_2', 'concat'),
                             ('concat', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': np.array([1, 3, 227, 227])},
                             'node_2': {'shape': np.array([1, 2, 227, 227])},
                             'concat': {'axis': 2}
                             })

        concat_node = Node(graph, 'concat')
        concat_infer(concat_node)
        res_shape = graph.node['node_3']['shape']
        self.assertIsNone(res_shape)

    def test_tf_concat_infer_no_shape(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'concat'),
                             ('node_2', 'concat'),
                             ('concat', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': np.array([1, 3, 227, 227])},
                             'node_2': {'shape': None},
                             'concat': {'axis': 2}
                             })

        concat_node = Node(graph, 'concat')
        concat_infer(concat_node)
        res_shape = graph.node['node_3']['shape']
        self.assertIsNone(res_shape)
