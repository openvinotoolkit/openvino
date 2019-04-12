"""
 Copyright (c) 2019 Intel Corporation

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

from extensions.ops.exp import ExpOp
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph

nodes_attributes = {'node_1': {'op': 'Identity', 'kind': 'op'},
                    'exp': {'op': 'Exp', 'kind': 'op'},
                    'node_3': {'op': 'Identity', 'kind': 'op'},
                    'op_output': {'kind': 'op', 'op': 'OpOutput'}
                    }


class TestExpOp(unittest.TestCase):
    def test_shape_only(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'exp'),
                             ('exp', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': np.array([1, 3, 10, 20])},
                             })

        exp_node = Node(graph, 'exp')
        ExpOp.infer(exp_node)
        exp_shape = np.array([1, 3, 10, 20])
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

    def test_shape_and_value(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'exp'),
                             ('exp', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {
                                'node_3': {
                                    'shape': None,
                                    'value': None,
                                },
                                'node_1': {
                                    'shape': np.array([2]),
                                    'value': np.array([0, 1], dtype=np.float32),
                                },
                            })

        exp_node = Node(graph, 'exp')
        ExpOp.infer(exp_node)
        exp_shape = np.array([2])
        exp_value = np.array([1, 2.7182818], dtype=np.float32)
        res_shape = graph.node['node_3']['shape']
        res_value = graph.node['node_3']['value']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])
        for i in range(0, len(exp_value)):
            self.assertAlmostEqual(exp_value[i], res_value[i], places=6)
