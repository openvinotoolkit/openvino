"""
 Copyright (C) 2020 Intel Corporation

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

from extensions.ops.Reverse import Reverse
from mo.front.common.extractors.utils import layout_attrs
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph

nodes_attributes = {'node_1': {'type': 'Identity', 'kind': 'op'},
                    'node_1_data': {'type': 'Identity', 'kind': 'data', 'value': np.array([[1, 3, 227, 227]])},
                    'node_2': {'type': 'Identity', 'kind': 'op'},
                    'node_2_data': {'kind': 'data', 'value': np.array([1])},
                    'reverse': {'type': 'Reverse', 'kind': 'op', },
                    'node_3': {'type': 'Identity', 'kind': 'op'},
                    'op_output': { 'kind': 'op', 'op': 'Result'}
                    }

class TestReverse(unittest.TestCase):
    def test_reverse_infer(self):
        graph = build_graph(nodes_attributes,
                            [
                             ('node_1', 'node_1_data'),
                             ('node_1_data', 'reverse'),
                             ('node_2', 'node_2_data'),
                             ('node_2_data', 'reverse'),
                             ('reverse', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None, 'value': None},
                             'node_1_data': {'shape': np.array([1, 4])},
                             'reverse': {'stride': 2,
                                       **layout_attrs()}
                             })

        reverse_node = Node(graph, 'reverse')
        Reverse.infer(reverse_node)
        exp_shape = np.array([1, 4])
        exp_value = np.array([[227, 227, 3, 1]])
        res_shape = graph.node['node_3']['shape']
        res_value = graph.node['node_3']['value']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])
        for i in range(0, len(exp_value[0])):
            self.assertEqual(exp_value[0][i], res_value[0][i])
