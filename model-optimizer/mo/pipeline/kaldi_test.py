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
from mo.pipeline.kaldi import apply_biases_to_last_layer
from mo.utils.unittest.graph import build_graph


class TestKaldiPipeline(unittest.TestCase):
    def test_apply_biases_to_ScaleShift(self):
        nodes = {'input': {'kind': 'data'},
                 'weights': {'value': None, 'kind': 'data'},
                 'biases': {'value': np.zeros(10), 'kind': 'data'},
                 'sc': {'op': 'ScaleShift', 'kind': 'op'},
                 'output': {'kind': 'data'},
                 'op_output': {'op': 'OpOutput', 'kind': 'op'}
                 }
        graph = build_graph(nodes,
                            [
                                ('input', 'sc'),
                                ('weights', 'sc'),
                                ('biases', 'sc'),
                                ('sc', 'output'),
                                ('output', 'op_output')
                            ])
        counts = -0.5 * np.ones(10)
        apply_biases_to_last_layer(graph, counts)
        sc_node = Node(graph, 'sc')
        self.assertTrue(np.array_equal(sc_node.in_node(2).value, -counts))

    def test_apply_biases_to_FullyConnected(self):
        nodes = {'input': {'kind': 'data'},
                 'weights': {'kind': 'data'},
                 'biases': {'value': None, 'shape': None, 'kind': 'data'},
                 'fc': {'op': 'FullyConnected', 'kind': 'op'},
                 'output': {'kind': 'data'},
                 'op_output': {'op': 'OpOutput', 'kind': 'op'}
                 }
        graph = build_graph(nodes,
                            [
                                ('input', 'fc'),
                                ('weights', 'fc'),
                                ('biases', 'fc'),
                                ('fc', 'output'),
                                ('output', 'op_output')
                            ])
        counts = -0.5 * np.ones(10)
        apply_biases_to_last_layer(graph, counts)
        fc_node = Node(graph, 'fc')
        self.assertTrue(np.array_equal(fc_node.in_node(2).value, -counts))

    def test_apply_biases_to_graph_with_SoftMax(self):
        nodes = {'input': {'kind': 'data'},
                 'weights': {'value': None, 'kind': 'data'},
                 'biases': {'value': None, 'shape': None, 'kind': 'data'},
                 'fc': {'op': 'FullyConnected', 'kind': 'op'},
                 'data': {'kind': 'data'},
                 'softmax': {'op': 'SoftMax', 'kind': 'op'},
                 'output': {'kind': 'data'},
                 'op_output': {'op': 'OpOutput', 'kind': 'op'}
                 }
        graph = build_graph(nodes,
                            [
                                ('input', 'fc'),
                                ('weights', 'fc'),
                                ('biases', 'fc'),
                                ('fc', 'data'),
                                ('data', 'softmax'),
                                ('softmax', 'output'),
                                ('output', 'op_output')
                            ])
        counts = -0.5 * np.ones(10)
        apply_biases_to_last_layer(graph, counts)
        fc_node = Node(graph, 'fc')
        self.assertTrue(np.array_equal(fc_node.in_node(2).value, -counts))
