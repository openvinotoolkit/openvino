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

from extensions.ops.merge import Merge
from mo.graph.graph import Node
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph_with_attrs


class TestMerge(unittest.TestCase):
    nodes = [
        ('first', {'value': np.ones((2, 2)), 'kind': 'data', 'executable': True, 'shape': np.array([2, 2]),
                   'is_partial_inferred': True}),
        ('second', {'value': np.zeros((2, 2)), 'kind': 'data', 'executable': False, 'shape': np.array([2, 2]),
                    'is_partial_inferred': True}),
        ('merge', {'type': 'Merge', 'kind': 'op', 'op': 'Merge'}),
        ('merge_output', {'value': None, 'kind': 'data', 'executable': True, 'shape': None}),
    ]
    edges = [
        ('first', 'merge', {'in': 0}),
        ('second', 'merge', {'in': 1}),
        ('merge', 'merge_output', {'out': 0}),
    ]

    def test_merge_infer_simple_case_one_executable(self):
        graph = build_graph_with_attrs(nodes_with_attrs=self.nodes, edges_with_attrs=self.edges)

        # We should propagate value of the first input since only this input is executable
        graph_ref = build_graph_with_attrs(nodes_with_attrs=self.nodes,
                                           edges_with_attrs=self.edges,
                                           update_nodes_attributes=[('merge_output', {'shape': np.array([2, 2]),
                                                                                      'value': np.ones((2,2))}),
                                                                    ('merge', {'is_not_fully_inferred': False})])

        tested_class = Merge(graph=graph, attrs={})
        node = Node(graph, 'merge')
        tested_class.merge_infer(node)

        (flag, resp) = compare_graphs(graph, graph_ref, 'merge_output', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_merge_infer_complex_case(self):
        """
        Case as in cycles when in first visit only one input are inferred and in the second -- both.
        """
        graph = build_graph_with_attrs(nodes_with_attrs=self.nodes, edges_with_attrs=self.edges,
                                       update_nodes_attributes=[('first', {'is_partial_inferred': False,
                                                                           'value': None}),
                                                                ('second', {'executable': True})])

        # In first visit we should propagate only shapes
        graph_ref = build_graph_with_attrs(nodes_with_attrs=self.nodes,
                                           edges_with_attrs=self.edges,
                                           update_nodes_attributes=[('second', {'executable': True}),
                                                                    ('first', {'is_partial_inferred': False,
                                                                                'value': None}),
                                                                    ('merge_output', {'shape': np.array([2, 2]),
                                                                                      'value': None}),
                                                                    ('merge', {'is_not_fully_inferred': True})])
        tested_class = Merge(graph=graph, attrs={})
        node = Node(graph, 'merge')
        tested_class.merge_infer(node)

        (flag, resp) = compare_graphs(graph, graph_ref, 'merge_output', check_op_attrs=True)
        self.assertTrue(flag, resp)

        # Imitate that inputs nodes now is inferred
        graph.node['first']['is_partial_inferred'] = True

        # Run infer second time
        tested_class = Merge(graph=graph, attrs={})
        node = Node(graph, 'merge')
        tested_class.merge_infer(node)

        graph_ref = build_graph_with_attrs(nodes_with_attrs=self.nodes,
                                           edges_with_attrs=self.edges,
                                           update_nodes_attributes=[('second', {'executable': True}),
                                                                    ('first', {'is_partial_inferred': True,
                                                                               'value': None}),
                                                                    ('merge_output', {'shape': np.array([2, 2]),
                                                                                      'value': None}),
                                                                    ('merge', {'is_not_fully_inferred': False})])
        (flag, resp) = compare_graphs(graph, graph_ref, 'merge_output', check_op_attrs=True)
        self.assertTrue(flag, resp)
