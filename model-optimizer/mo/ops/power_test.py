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
from mo.ops.power import Power
from mo.utils.unittest.graph import build_graph


class TestPowerOp(unittest.TestCase):
    @staticmethod
    def create_graph(single_input=True):
        nodes_attributes = {
            'input1': {
                'kind': 'data',
                'shape': np.array([1, 3, 224, 224]),
                'value': None,
            },
            'input2': {
                'kind': 'data',
                'shape': np.array([]),
                'value': np.array(1.0),
            },
            'power': {
                'kind': 'op',
                'shape': np.array([1, 3, 224, 224]),
            },
            'power_data': {
                'kind': 'data',
                'shape': None,
            },
        }
        if single_input:
            return build_graph(nodes_attributes,
                               [
                                   ('input1', 'power'),
                                   ('power', 'power_data')
                               ])
        else:
            return build_graph(nodes_attributes,
                               [
                                   ('input1', 'power'),
                                   ('input2', 'power'),
                                   ('power', 'power_data')
                               ])

    def test_power_single_input_infer1(self):
        graph = self.create_graph(single_input=True)
        graph.graph['layout'] = 'NCHW'
        power_node = Node(graph, 'power')
        power_node['power'] = 1.0

        Power.infer(power_node)

        self.assertTrue(np.array_equal(power_node.out_node().shape, power_node.in_node(0).shape))

    def test_power_two_input_infer1(self):
        graph = self.create_graph(single_input=False)
        graph.graph['layout'] = 'NCHW'
        power_node = Node(graph, 'power')

        Power.infer(power_node)

        self.assertTrue(np.array_equal(power_node.out_node().shape, power_node.in_node(0).shape))

    def test_power_two_input_infer2(self):
        graph = self.create_graph(single_input=False)
        power_node = Node(graph, 'power')
        input2 = Node(graph, 'input2')
        input2.value = np.ones((1, 2, 3))

        Power.infer(power_node)

        self.assertIsNone(power_node.out_node().shape)

    def test_power_two_input_infer3(self):
        graph = self.create_graph(single_input=False)
        power_node = Node(graph, 'power')
        input2 = Node(graph, 'input2')
        input2.value = None

        Power.infer(power_node)

        self.assertIsNone(power_node.out_node().shape)
