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

import itertools
import unittest

import numpy as np
from generator import generator, generate

from mo.graph.graph import Node
from mo.ops.permute import Permute
from mo.utils.unittest.graph import build_graph


@generator
class TestPermuteOp(unittest.TestCase):
    nodes_attributes = {
        'data_1': {
            'kind': 'data',
            'shape': np.array([1, 3, 224, 224])
        },
        'transpose': {
            'type': 'Permute',
            'order': None,
            'reverse_order': False,
            'kind': 'op',
        },
        'data_2': {
            'kind': 'data',
            'shape': None,
        }
    }

    def _create_graph_with_transpose(self, order):
        graph = build_graph(self.nodes_attributes,
                            [('data_1', 'transpose'),
                             ('transpose', 'data_2')],
                            {'transpose': {'order': order}})
        return graph

    @generate(*[list(order) for order in list(itertools.permutations(np.arange(4)))])
    def test_transpose_infer_1(self, order):
        graph = self._create_graph_with_transpose(order)
        transpose_node = Node(graph, 'transpose')

        Permute.infer(transpose_node)

        ref = [transpose_node.in_node().shape[i] for i in order]
        self.assertTrue(np.array_equal(transpose_node.out_node().shape, np.array(ref)))

    def test_transpose_infer_2(self):
        order = None
        graph = self._create_graph_with_transpose(order)
        transpose_node = Node(graph, 'transpose')
        transpose_node['reverse_order'] = True

        Permute.infer(transpose_node)

        ref = np.array([x for x in reversed(transpose_node.in_node().shape)])
        self.assertTrue(np.array_equal(transpose_node.out_node().shape, ref),
                        "Shapes are not the same: {} and {}".format(transpose_node.out_node().shape, ref))

    def test_transpose_infer_neg_1(self):
        order = np.array([0, 1, 2, 3])
        graph = self._create_graph_with_transpose(order)
        transpose_node = Node(graph, 'transpose')
        transpose_node['reverse_order'] = True

        Permute.infer(transpose_node)

        ref = None
        self.assertTrue(transpose_node.out_node().shape is None, "Output shape should be None")

    def test_transpose_infer_neg_2(self):
        order = None
        graph = self._create_graph_with_transpose(order)
        transpose_node = Node(graph, 'transpose')
        transpose_node['reverse_order'] = False

        Permute.infer(transpose_node)

        ref = None
        self.assertTrue(transpose_node.out_node().shape is None, "Output shape should be None")
