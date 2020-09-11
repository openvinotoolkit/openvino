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

from mo.front.common.partial_infer.roipooling import roipooling_infer
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph

nodes_attributes = {'node_1': {'kind': 'data'},
                    'node_2': {'kind': 'data'},
                    'node_3': {'kind': 'data'},
                    'node_4': {'kind': 'data'},
                    'roipool': {'type': 'ROIPooling', 'kind': 'op', 'pooled_h': None, 'pooled_w': None},
                    'output': {'value': None, 'kind': 'data'},
                    'op_output': { 'kind': 'op', 'op': 'Result'},
                    }


class TestRoipoolingInfer(unittest.TestCase):
    def test_roipooling_infer_ideal(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'roipool'),
                             ('node_2', 'roipool'),
                             ('roipool', 'output'),
                             ('output', 'op_output')
                             ],
                            {'output': {'shape': None},
                             'node_1': {'shape': np.array([1, 256, 20, 20])},
                             'node_2': {'shape': np.array([150, 5])},
                             'roipool': {'pooled_h': 6, 'pooled_w': 6}
                             })
        graph.graph['layout'] = 'NCHW'
        roipooling_node = Node(graph, 'roipool')

        roipooling_infer(roipooling_node)
        exp_shape = np.array([150, 256, 6, 6])
        res_shape = graph.node['output']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

    def test_roipooling_infer_no_shape(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'roipool'),
                             ('node_2', 'roipool'),
                             ('roipool', 'output'),
                             ('output', 'op_output')
                             ],
                            {'output': {'shape': None},
                             'node_1': {'shape': None},
                             'node_2': {'shape': np.array([1, 256])},
                             'roipool': {'pooled_h': 6, 'pooled_w': 6}
                             })
        graph.graph['layout'] = 'NCHW'

        roipooling_node = Node(graph, 'roipool')

        roipooling_infer(roipooling_node)
        self.assertIsNone(graph.node['output']['shape'])

    def test_roipooling_infer_tf(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'roipool'),
                             ('node_2', 'roipool'),
                             ('node_3', 'roipool'),
                             ('node_4', 'roipool'),
                             ('roipool', 'output'),
                             ('output', 'op_output')
                             ],
                            {'output': {'shape': None},
                             'node_1': {'shape': np.array([1, 20, 20, 256])},
                             'node_2': {'shape': np.array([150, 5])},
                             'node_3': {'shape': np.array([150])},
                             'node_4': {'shape': np.array([2], dtype=np.int64), 'value': np.array([7, 6],
                                                                                                  dtype=np.int64)},
                             })
        graph.graph['layout'] = 'NHWC'
        roipooling_node = Node(graph, 'roipool')

        roipooling_infer(roipooling_node)
        exp_shape = np.array([150, 7, 6, 256])
        res_shape = graph.node['output']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])
