# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.middle.TensorIteratorInput import SmartInputMatcher, SimpleInputMatcher, BackEdgeSimpleInputMatcher
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph_with_attrs


class SmartInputMatcherTests(unittest.TestCase):
    def test(self):
        pattern_matcher = SmartInputMatcher()
        pattern = pattern_matcher.pattern()

        graph = build_graph_with_attrs(nodes_with_attrs=pattern['nodes'], edges_with_attrs=pattern['edges'],
                                       update_edge_attrs={('range_data', 'TensorArrayScatter', 0): {'in': 1},
                                                          ('TensorArray_handle', 'TensorArrayScatter', 0): {'in': 0},
                                                          ('TensorArray_flow', 'TensorArrayScatter', 0): {'in': 3}},
                                       new_nodes_with_attrs=[('ta_size', {'kind': 'data'}),
                                                             ('ta_size_op', {'kind': 'op'}),
                                                             ('value', {'kind': 'data'}),
                                                             ],
                                       new_edges_with_attrs=[
                                           ('ta_size_op', 'ta_size'),
                                           ('ta_size', 'TensorArray'),
                                           ('value', 'TensorArrayScatter', {'in':2}),
                                                             ],
                                       update_nodes_attributes=[('Enter_data', {'value': np.array([1])}),
                                                                ('stack_data', {'value': np.array([0])}),
                                                                ('stack_1_data', {'value': np.array([1])}),
                                                                ('stack_2_data', {'value': np.array([1])}),
                                                                ('start_data', {'value': np.array([0])}),
                                                                ('delta_data', {'value': np.array([1])})
                                                                ])

        pattern_matcher.find_and_replace_pattern(graph)
        graph_ref = build_graph_with_attrs(
            nodes_with_attrs=[('condition_data', {'kind': 'data'}),
                              ('TensorIteratorInput', {'kind': 'op', 'op': 'TensorIteratorInput'}),
                              ('TensorArrayRead_data', {'kind': 'data'}),
                              ('condition_data', {'kind': 'data'}),
                              ('value', {'kind': 'data'}),
                              ('ta_size', {'kind': 'data'}),
                              ('ta_size_op', {'kind': 'op'})],
            edges_with_attrs=[('ta_size', 'TensorIteratorInput', {'in': 0}),
                              ('condition_data', 'TensorIteratorInput', {'in': 2}),
                              ('value', 'TensorIteratorInput', {'in': 1}),
                              ('TensorIteratorInput', 'TensorArrayRead_data'),
                              ('ta_size_op', 'ta_size')],
            update_edge_attrs=None,
            new_nodes_with_attrs=[],
            new_edges_with_attrs=[],
            )
        (flag, resp) = compare_graphs(graph, graph_ref, 'TensorArrayRead_data', check_op_attrs=True)
        self.assertTrue(flag, resp)


class SimpleInputMatcherTest(unittest.TestCase):
    def test(self):
        pattern_matcher = SimpleInputMatcher()
        pattern = pattern_matcher.pattern()

        graph = build_graph_with_attrs(nodes_with_attrs=pattern['nodes'], edges_with_attrs=pattern['edges'],
                                       update_edge_attrs=None,
                                       new_nodes_with_attrs=[('in_node', {'kind': 'data'}),
                                                             ('Enter_data', {'kind': 'data'})],
                                       new_edges_with_attrs=[('in_node', 'Enter'), ('Enter', 'Enter_data')],
                                       update_nodes_attributes=[])

        pattern_matcher.find_and_replace_pattern(graph)

        graph_ref = build_graph_with_attrs(
            nodes_with_attrs=[('TensorIteratorInput', {'kind': 'op', 'op': 'TensorIteratorInput'}),
                              ('in_node', {'kind': 'data'}),
                              ('Enter_data', {'kind': 'data'})
                              ],
            edges_with_attrs=[('in_node', 'TensorIteratorInput'), ('TensorIteratorInput', 'Enter_data')],
        )
        (flag, resp) = compare_graphs(graph, graph_ref, 'Enter_data', check_op_attrs=True)
        self.assertTrue(flag, resp)


class BackEdgeInputMatcherTest(unittest.TestCase):
    def test1(self):
        """
        Case with constant input to init
        """
        pattern_matcher = BackEdgeSimpleInputMatcher()
        pattern = pattern_matcher.pattern()

        graph = build_graph_with_attrs(nodes_with_attrs=pattern['nodes'], edges_with_attrs=pattern['edges'],
                                       new_nodes_with_attrs=[('cycle_data', {'kind': 'data'}),
                                                             ('condition', {'kind': 'data'}),
                                                             ('init', {'kind': 'data', 'shape': np.array([1,3])}),
                                                             ],
                                       new_edges_with_attrs=[('condition', 'BackEdge', {'in': 2}),
                                                             ('init', 'BackEdge', {'in': 0}),
                                                             ('cycle_data', 'BackEdge', {'in': 1})],)

        pattern_matcher.find_and_replace_pattern(graph)
        graph_ref = build_graph_with_attrs(nodes_with_attrs=pattern['nodes'], edges_with_attrs=pattern['edges'],
                                       new_nodes_with_attrs=[('cycle_data', {'kind': 'data'}),
                                                             ('condition', {'kind': 'data'}),
                                                             ('init', {'kind': 'data', 'shape': np.array([1,3])}),
                                                             ('TensorIteratorInput', {'kind': 'op', 'op': 'TensorIteratorInput'}),
                                                             ('TensorIteratorInput_data', {'kind': 'data', 'shape': np.array([1,3])}),
                                                             ],
                                       new_edges_with_attrs=[('TensorIteratorInput_data', 'TensorIteratorInput'),
                                                             ('TensorIteratorInput', 'init'),
                                                            ('condition', 'BackEdge', {'in': 2}),
                                                             ('init', 'BackEdge', {'in': 0}),
                                                             ('cycle_data', 'BackEdge', {'in': 1})],)
        (flag, resp) = compare_graphs(graph, graph_ref, 'BackEdge', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test2(self):
        """
        Case with non-constant input to init.
        Nothing should happen with graph.
        """
        pattern_matcher = BackEdgeSimpleInputMatcher()
        pattern = pattern_matcher.pattern()

        graph = build_graph_with_attrs(nodes_with_attrs=pattern['nodes'], edges_with_attrs=pattern['edges'],
                                       new_nodes_with_attrs=[('cycle_data', {'kind': 'data'}),
                                                             ('condition', {'kind': 'data'}),
                                                             ('init', {'kind': 'data', 'shape': np.array([1, 3])}),
                                                             ('Enter', {'kind': 'op', 'op': 'Enter'}),
                                                             ],
                                       new_edges_with_attrs=[('Enter', 'init'),
                                                             ('condition', 'BackEdge', {'in': 2}),
                                                             ('init', 'BackEdge', {'in': 0}),
                                                             ('cycle_data', 'BackEdge', {'in': 1})])

        pattern_matcher.find_and_replace_pattern(graph)

        graph_ref = build_graph_with_attrs(nodes_with_attrs=pattern['nodes'], edges_with_attrs=pattern['edges'],
                                       new_nodes_with_attrs=[('cycle_data', {'kind': 'data'}),
                                                             ('condition', {'kind': 'data'}),
                                                             ('init', {'kind': 'data', 'shape': np.array([1, 3])}),
                                                             ('Enter', {'kind': 'op', 'op': 'Enter'}),
                                                             ],
                                       new_edges_with_attrs=[('Enter', 'init'),
                                                             ('condition', 'BackEdge', {'in': 2}),
                                                             ('init', 'BackEdge', {'in': 0}),
                                                             ('cycle_data', 'BackEdge', {'in': 1})], )

        (flag, resp) = compare_graphs(graph, graph_ref, 'BackEdge', check_op_attrs=True)
        self.assertTrue(flag, resp)
