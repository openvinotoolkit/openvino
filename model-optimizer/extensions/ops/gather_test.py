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

from extensions.ops.gather import Gather
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph


class TestGatherPartialInfer(unittest.TestCase):
    @staticmethod
    def _create_graph():
        nodes_attributes = {'gather_input': {'shape': None, 'value': None, 'kind': 'data'},
                            'gather_input2': {'shape': None, 'value': None, 'kind': 'data'},
                            'gather_node': {'op': 'Gather', 'kind': 'op'},
                            'gather_output': {'shape': None, 'value': None, 'kind': 'data'}
                            }
        return build_graph(nodes_attributes,
                           [
                               ('gather_input', 'gather_node'), ('gather_node', 'gather_output'), ('gather_input2', 'gather_node')
                           ],
                           {
                               'gather_input': {'shape': int64_array([10, 15]), 'value': np.ones((3, 15))},
                               'gather_input2': {'shape': int64_array([2]), 'value': np.array([0, 2])},
                               'gather_node': {'axis': 0},
                           })

    def test_gather_infer(self):
        graph = self._create_graph()

        gather_node = Node(graph, 'gather_node')
        Gather.infer(gather_node)

        exp_shape = int64_array([2, 15])
        res_shape = graph.node['gather_output']['shape']
        res_value = graph.node['gather_output']['value']

        self.assertTrue(np.array_equal(exp_shape, res_shape),
                        'shapes do not match expected: {} and given: {}'.format(exp_shape, res_shape))

        self.assertTrue(np.array_equal(res_value, np.ones(exp_shape)),
                        'shapes do not match expected: {} and given: {}'.format(exp_shape, res_shape))
