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

from extensions.front.kaldi.add_permute_after_convolution import ReplaceConvolutionTranspose
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph


class ReplaceConvolutionTransposeTests(unittest.TestCase):
    nodes_attributes = {
        'conv': {'kind': 'op', 'op': 'Convolution'},
        'reshape_conv': {'kind': 'op', 'op': 'Reshape'},
        'reshape_pool': {'kind': 'op', 'op': 'Reshape'},
        'pool': {'kind': 'op', 'op': 'Pooling'},
        'reshape_after_pool': {'kind': 'op', 'op': 'Reshape'},
        'act': {'kind': 'op', 'op': 'Sigmoid'},
        'fc': {'kind': 'op', 'op': 'FullyConnected'},
        'scale_shift': {'kind': 'op', 'op': 'ScaleShift'}
    }

    def test_simple_convolution(self):
        graph = build_graph(self.nodes_attributes, [
            ('conv', 'reshape_conv'),
            ('reshape_conv', 'scale_shift'),
        ])
        ReplaceConvolutionTranspose().find_and_replace_pattern(graph)
        conv_node = Node(graph, graph.nodes['conv']['name'])
        permute = conv_node.out_node()
        self.assertEqual(permute.op, 'Transpose')
        self.assertTrue(np.array_equal(permute.in_node(1).in_node().value, np.array([0, 3, 2, 1])))

    def test_conv_pool(self):
        graph = build_graph(self.nodes_attributes, [
            ('conv', 'reshape_conv'),
            ('reshape_conv', 'reshape_pool'),
            ('reshape_pool', 'pool'),
            ('pool', 'reshape_after_pool'),
            ('reshape_after_pool', 'fc'),
        ])
        ReplaceConvolutionTranspose().find_and_replace_pattern(graph)
        pool_node = Node(graph, graph.nodes['pool']['name'])
        permute = pool_node.out_node()
        self.assertEqual(permute.op, 'Transpose')
        self.assertTrue(np.array_equal(permute.in_node(1).in_node().value, np.array([0, 3, 2, 1])))

    def test_conv_act_pool(self):
        graph = build_graph(self.nodes_attributes, [
            ('conv', 'reshape_conv'),
            ('reshape_conv', 'act'),
            ('act', 'reshape_pool'),
            ('reshape_pool', 'pool'),
            ('pool', 'reshape_after_pool'),
            ('reshape_after_pool', 'fc'),
        ])
        ReplaceConvolutionTranspose().find_and_replace_pattern(graph)
        pool_node = Node(graph, graph.nodes['pool']['name'])
        permute = pool_node.out_node()
        self.assertEqual(permute.op, 'Transpose')
        self.assertTrue(np.array_equal(permute.in_node(1).in_node().value, np.array([0, 3, 2, 1])))
