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

from extensions.ops.elementwise import Round
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph

def round_test_graph(nodes_attributes, value, mode: str):
    graph = build_graph(nodes_attributes,
                        [
                            ('node_1', 'elementwise_node'),
                            ('elementwise_node', 'node_3')
                        ],
                        {
                            'node_1': {
                                'value': value
                            },
                            'elementwise_node': {
                                'op': 'Round',
                                'mode': mode,
                            },
                            'node_3': {
                                'value': None
                            }
                        })
    return graph


class TestElementwiseOp(unittest.TestCase):
    nodes_attributes = {
        'node_1': {
            'shape': np.array([13]),
            'value': None
        },
        'elementwise_node': {
            'op': None,
            'kind': 'op',
            'operation': None
        },
        'node_3': {
            'shape': None
        }
    }

    value = np.array([-23.5, -22.5, -2.5, -1.5, -0.5, 0.5, 0.9, 1.5, 2.3, 2.5, 3.5, 22.5, 23.5])

    def test_elementwise_round_even_infer(self):
        graph = round_test_graph(self.nodes_attributes, self.value, 'half_to_even')

        graph.graph['layout'] = 'NCHW'
        elementwise_node = Node(graph, 'elementwise_node')
        Round.infer(elementwise_node)
        exp_shape = np.array([13])
        res_shape = graph.node['node_3']['shape']
        res_value = graph.node['node_3']['value']
        exp_value = np.array([-24., -22., -2., -2., -0., 0., 1., 2., 2., 2., 4., 22., 24.,])
        for i, value in enumerate(exp_shape):
            self.assertEqual(res_shape[i], value)
        for i, value in enumerate(exp_value):
            self.assertAlmostEqual(res_value[i], value)

    def test_elementwise_round_away_infer(self):
        graph = round_test_graph(self.nodes_attributes, self.value, 'half_away_from_zero')

        graph.graph['layout'] = 'NCHW'
        elementwise_node = Node(graph, 'elementwise_node')
        Round.infer(elementwise_node)
        exp_shape = np.array([13])
        res_shape = graph.node['node_3']['shape']
        res_value = graph.node['node_3']['value']
        exp_value = np.array([-24., -23., -3., -2., -1., 1., 1., 2., 2., 3., 4., 23., 24.])
        for i, value in enumerate(exp_shape):
            self.assertEqual(res_shape[i], value)
        for i, value in enumerate(exp_value):
            self.assertAlmostEqual(res_value[i], value)
