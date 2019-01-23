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

from mo.graph.graph import Node
from mo.ops.pad import Pad
from mo.utils.unittest.graph import build_graph


class TestPadONNXOp(unittest.TestCase):
    # There are tests for InnerProduct.infer in mo/front/common/partial_infer/inner_product_test.py
    node_attrs = {
        'data_in': {
            'kind': 'data',
            'shape': np.array([1, 3, 100, 200])
        },
        # optional input for one of the two flavors of pad op
        'data_pads': {
            'kind': 'data',
            'value': np.array([[0, 0], [0, 0], [1, 3], [2, 4]], dtype=np.int64),
            'shape': np.array([2, 4], dtype=np.int64)
        },
        'pad': {
            'op': 'Pad',
            'kind': 'op',
            'pads': None,
        },
        'data_out': {
            'kind': 'data',
            'shape': None,
        }
    }

    edge_attrs = [
        ('data_in', 'pad'),
        ('pad', 'data_out')
    ]

    def test_one_input(self):
        graph = build_graph(
            self.node_attrs,
            self.edge_attrs,
            {'pad': {'pads': np.array([[0, 0], [0, 0], [1, 3], [2, 4]], dtype=np.int64)}},
            nodes_with_edges_only=True,
        )
        pad_node = Node(graph, 'pad')
        Pad.infer(pad_node)
        self.assertTrue(np.array_equal(Node(graph, 'data_out').shape, np.array([1, 3, 100 + 1 + 3, 200 + 2 + 4])))

    def test_two_inputs(self):
        graph = build_graph(
            self.node_attrs,
            self.edge_attrs + [('data_pads', 'pad')],
            nodes_with_edges_only=True,
        )
        pad_node = Node(graph, 'pad')
        Pad.infer(pad_node)
        self.assertTrue(np.array_equal(Node(graph, 'data_out').shape, np.array([1, 3, 100 + 1 + 3, 200 + 2 + 4])))

    def test_one_input_and_no_pads(self):
        graph = build_graph(
            self.node_attrs,
            self.edge_attrs,
            nodes_with_edges_only=True,
        )
        pad_node = Node(graph, 'pad')
        with self.assertRaisesRegex(AssertionError, ".*pads attribute is missing.*"):
            Pad.infer(pad_node)

    def test_two_inputs_and_pads(self):
        graph = build_graph(
            self.node_attrs,
            self.edge_attrs + [('data_pads', 'pad')],
            {'pad': {'pads': np.array([[0, 0], [0, 0], [1, 3], [2, 4]], dtype=np.int64)}},
            nodes_with_edges_only=True,
        )
        pad_node = Node(graph, 'pad')
        with self.assertRaisesRegex(AssertionError, ".*unexpected additional input argument.*"):
            Pad.infer(pad_node)
