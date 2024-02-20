# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.ops.merge import Merge
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph_with_attrs


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
                                                                                      'value': np.ones((2, 2))}),
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

    def test_merge_infer_only_second_executable(self):
        graph = build_graph_with_attrs(
            nodes_with_attrs=self.nodes,
            edges_with_attrs=self.edges,
            update_nodes_attributes=[
                ('first', {'executable': False, 'value': np.ones([2, 2]), 'shape': int64_array([2, 2])}),
                ('second', {'executable': True, 'value': np.zeros([4, 4]), 'shape': int64_array([4, 4])})
            ]
        )

        ref_graph = build_graph_with_attrs(
            nodes_with_attrs=self.nodes,
            edges_with_attrs=self.edges,
            update_nodes_attributes=[
                ('first', {'executable': False, 'value': np.ones([2, 2]), 'shape': int64_array([2, 2])}),
                ('second', {'executable': True, 'value': np.zeros([4, 4]), 'shape': int64_array([4, 4])}),
                ('merge', {'is_not_fully_inferred': False}),
                ('merge_output', {'shape': int64_array([4, 4]), 'value': np.zeros([4, 4])})
            ]
        )

        tested_class = Merge(graph=graph, attrs={})
        node = Node(graph, 'merge')
        tested_class.merge_infer(node)

        (flag, resp) = compare_graphs(graph, ref_graph, 'merge_output', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_merge_infer_no_executable(self):
        graph = build_graph_with_attrs(
            nodes_with_attrs=self.nodes,
            edges_with_attrs=self.edges,
            update_nodes_attributes=[
                ('first', {'executable': False, 'value': np.ones([2, 2]), 'shape': int64_array([2, 2])}),
                ('second', {'executable': False, 'value': np.zeros([4, 4]), 'shape': int64_array([4, 4])})
            ]
        )

        ref_graph = build_graph_with_attrs(
            nodes_with_attrs=self.nodes,
            edges_with_attrs=self.edges,
            update_nodes_attributes=[
                ('first', {'executable': False, 'value': np.ones([2, 2]), 'shape': int64_array([2, 2])}),
                ('second', {'executable': False, 'value': np.zeros([4, 4]), 'shape': int64_array([4, 4])}),
                ('merge', {'is_not_fully_inferred': False}),
                ('merge_output', {'shape': int64_array([2, 2]), 'value': None})
            ]
        )

        tested_class = Merge(graph=graph, attrs={})
        node = Node(graph, 'merge')
        tested_class.merge_infer(node)

        (flag, resp) = compare_graphs(graph, ref_graph, 'merge_output', check_op_attrs=True)
        self.assertTrue(flag, resp)
