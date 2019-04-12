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

from extensions.ops.select import Select
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph_with_attrs, compare_graphs


class TestSelect(unittest.TestCase):
    nodes = [
        ('than', {'value': np.ones((2, 2)), 'kind': 'data', 'executable': True, 'shape': np.array([2, 2])}),
        ('else', {'value': np.zeros((2, 2)), 'kind': 'data', 'executable': True, 'shape': np.array([2, 2])}),
        ('condition', {'value': None, 'kind': 'data', 'executable': True}),
        ('select', {'type': 'Select', 'kind': 'op', 'op': 'Select'}),
        ('select_output', {'value': None, 'kind': 'data', 'executable': True, 'shape': None}),
    ]
    edges = [
        ('condition', 'select', {'in': 0}),
        ('than', 'select', {'in': 1}),
        ('else', 'select', {'in': 2}),
        ('select', 'select_output', {'out': 0}),
    ]

    def test_select_infer_no_condition(self):
        graph = build_graph_with_attrs(nodes_with_attrs=self.nodes, edges_with_attrs=self.edges)

        # We should propagate only shapes
        graph_ref = build_graph_with_attrs(nodes_with_attrs=self.nodes,
                                           edges_with_attrs=self.edges,
                                           update_nodes_attributes=[('select_output', {'shape': np.array([2, 2])})])

        tested_class = Select(graph=graph, attrs={})

        node = Node(graph, 'select')
        tested_class.infer(node)

        (flag, resp) = compare_graphs(graph, graph_ref, 'select_output', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_select_infer_condition_true(self):
        graph = build_graph_with_attrs(nodes_with_attrs=self.nodes, edges_with_attrs=self.edges,
                                       update_nodes_attributes=[('condition', {'value': np.array([True])})])

        # We should propagate shapes and values
        graph_ref = build_graph_with_attrs(nodes_with_attrs=self.nodes,
                                           edges_with_attrs=self.edges,
                                           update_nodes_attributes=[('select_output', {'shape': np.array([2, 2]),
                                                                                       'value': np.ones((2,2))})])

        tested_class = Select(graph=graph, attrs={})

        node = Node(graph, 'select')
        tested_class.infer(node)

        (flag, resp) = compare_graphs(graph, graph_ref, 'select_output', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_select_infer_condition_false(self):
        graph = build_graph_with_attrs(nodes_with_attrs=self.nodes, edges_with_attrs=self.edges,
                                       update_nodes_attributes=[('condition', {'value': np.array([False])})])

        # We should propagate shapes and values
        graph_ref = build_graph_with_attrs(nodes_with_attrs=self.nodes,
                                           edges_with_attrs=self.edges,
                                           update_nodes_attributes=[('select_output', {'shape': np.array([2, 2]),
                                                                                       'value': np.zeros((2, 2))})])

        tested_class = Select(graph=graph, attrs={})

        node = Node(graph, 'select')
        tested_class.infer(node)

        (flag, resp) = compare_graphs(graph, graph_ref, 'select_output', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_select_infer_assert_shapes(self):
        graph = build_graph_with_attrs(nodes_with_attrs=self.nodes, edges_with_attrs=self.edges,
                                       update_nodes_attributes=[('else', {'shape': np.array([3,3]), 'value':np.zeros((3,3))})])

        tested_class = Select(graph=graph, attrs={})

        node = Node(graph, 'select')
        with self.assertRaisesRegex(AssertionError, "TensorFlow \'Select\' operation has 3 inputs: \'condition\',"
                                                    " \'then\' and \'else\' tensors.\'then\' and \'else\' tensors"
                                                    " must have the same shape by TensorFlow reference"):
            tested_class.infer(node)

    def test_select_infer_assert_condition_bool(self):
        graph = build_graph_with_attrs(nodes_with_attrs=self.nodes, edges_with_attrs=self.edges,
                                       update_nodes_attributes=[('condition', {'value': np.array([3])})])

        tested_class = Select(graph=graph, attrs={})

        node = Node(graph, 'select')
        with self.assertRaisesRegex(AssertionError, "TensorFlow \'Select\' operation has 3 inputs: \'condition\',"
                                                    " \'then\' and \'else\' tensors. Value of \'condition\' tensor"
                                                    " must be boolen by TensorFlow reference"):
            tested_class.infer(node)