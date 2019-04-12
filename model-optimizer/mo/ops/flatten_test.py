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

from mo.graph.graph import Node
from mo.ops.flatten import Flatten
from mo.utils.unittest.graph import build_graph

nodes_attributes = {'node_1': {'value': None, 'kind': 'data'},
                    'flatten_1': {'type': 'Flatten', 'value': None, 'kind': 'op'},
                    'node_2': {'value': None, 'kind': 'data'},
                    'output_op': { 'kind': 'op', 'op': 'OpOutput'},
                    }


class TestFlattenPartialInfer(unittest.TestCase):
    def test_flatten_infer(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'flatten_1'),
                             ('flatten_1', 'node_2'),
                             ('node_2', 'op_output')
                             ],
                            {'node_2': {'shape': np.array([1, 3 * 256 * 256])},
                             'node_1': {'shape': np.array([1, 3, 256, 256])},
                             'flatten_1': {'axis': 1, 'dim': []}
                             })

        flatten_node = Node(graph, 'flatten_1')

        Flatten.infer(flatten_node)
        exp_shape = np.array([1, 3 * 256 * 256])
        res_shape = graph.node['node_2']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

    def test_flatten_infer_no_shape(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'flatten_1'),
                             ('flatten_1', 'node_2'),
                             ('node_2', 'op_output')
                             ],
                            {'node_2': {'shape': None},
                             'node_1': {'shape': None},
                             'flatten_1': {'axis': 1}
                             })

        flatten_node = Node(graph, 'flatten_1')

        Flatten.infer(flatten_node)
        res_shape = graph.node['node_2']['shape']
        self.assertIsNone(res_shape)
