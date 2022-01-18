# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.middle.TensorIteratorCondition import LoopConditionMatcher
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph_with_attrs, regular_op_with_empty_data, connect, build_graph


class TensorIteratorConditionTests(unittest.TestCase):
    def test_not_dynamic_1(self):
        pattern_matcher = LoopConditionMatcher()
        pattern = pattern_matcher.pattern(1)

        graph = build_graph_with_attrs(nodes_with_attrs=pattern['nodes'], edges_with_attrs=pattern['edges'],
                                       new_nodes_with_attrs=[
                                           ('maximum', {'kind': 'op', 'op': 'Maximum'}),
                                           ('maximum_data', {'kind': 'data'}),
                                           ('TensorIteratorInput', {'kind': 'op', 'op': 'TensorIteratorInput'}),
                                           ('some_op', {'kind': 'op', 'op': 'Add'}),
                                           ('minimum', {'kind': 'op', 'op': 'Minimum'})],
                                       new_edges_with_attrs=[
                                           ('Identity_1_data', 'TensorIteratorInput'),
                                           ('maximum', 'maximum_data'),
                                           ('maximum_data', 'minimum'),
                                           ('minimum', 'minimum_data'),
                                           ('loop_cond_data', 'some_op')],
                                       update_nodes_attributes=[
                                           ('init_1_data', {'value': np.array([0])}),
                                           ('init_2_data', {'value': np.array([0])}),
                                           ('add_1_y_data', {'value': np.array(1)}),
                                           ('add_2_y_data', {'value': np.array(1)}),
                                           ('loop_cond_data', {'value': None}),
                                           ('Identity_2_data', {'value': None},),
                                           ('Enter_1_less_data', {'value': None},),
                                       ])

        pattern_matcher.find_and_replace_pattern(graph)
        nodes_attributes = {
            **regular_op_with_empty_data('loop_cond', {'op': 'TensorIteratorCondition', 'type': None}),
            **regular_op_with_empty_data('StridedSlice', {'op': 'StridedSlice', 'type': None}),
            **regular_op_with_empty_data('maximum', {'op': 'Maximum', 'type': None}),
            **regular_op_with_empty_data('minimum', {'op': 'Minimum', 'type': None}),
            'some_op': {'kind': 'op', 'op': 'Add'},
            'identity_data': {'kind': 'data'},
            'TensorIteratorInput': {'kind': 'op', 'op': 'TensorIteratorInput'}
        }
        edges = [
            *connect('StridedSlice', '0:loop_cond'),
            *connect('maximum', 'minimum'),
            *connect('minimum', '1:loop_cond'),
            *connect('loop_cond', 'some_op'),
            ('loop_cond', 'identity_data'),
            ('identity_data', 'TensorIteratorInput')
        ]
        graph_ref = build_graph(nodes_attributes, edges)
        (flag, resp) = compare_graphs(graph, graph_ref, 'some_op', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_not_dynamic_2(self):
        pattern_matcher = LoopConditionMatcher()
        pattern = pattern_matcher.pattern(2)

        graph = build_graph_with_attrs(nodes_with_attrs=pattern['nodes'], edges_with_attrs=pattern['edges'],
                                       new_nodes_with_attrs=[
                                           ('TensorIteratorInput', {'kind': 'op', 'op': 'TensorIteratorInput'}),
                                           ('some_op', {'kind': 'op', 'op': 'Add'})],
                                       new_edges_with_attrs=[
                                           ('Identity_1_data', 'TensorIteratorInput'),
                                           ('loop_cond_data', 'some_op'),
                                       ],
                                       update_nodes_attributes=[
                                           ('init_1_data', {'value': np.array([0])}),
                                           ('init_2_data', {'value': np.array([0])}),
                                           ('add_1_y_data', {'value': np.array(1)}),
                                           ('add_2_y_data', {'value': np.array(1)}),
                                           ('loop_cond_data', {'value': None}),
                                           ('Identity_2_data', {'value': None},),
                                           ('Enter_1_less_data', {'value': None},),
                                       ])

        pattern_matcher.find_and_replace_pattern(graph)
        nodes_attributes = {
            **regular_op_with_empty_data('loop_cond', {'op': 'TensorIteratorCondition', 'type': None}),
            **regular_op_with_empty_data('StridedSlice', {'op': 'StridedSlice', 'type': None}),
            'some_op': {'kind': 'op', 'op': 'Add'},
            'identity_data': {'kind': 'data'},
            'TensorIteratorInput': {'kind': 'op', 'op': 'TensorIteratorInput'}
        }
        edges = [
            *connect('StridedSlice', 'loop_cond'),
            *connect('loop_cond', 'some_op'),
            ('loop_cond', 'identity_data'),
            ('identity_data', 'TensorIteratorInput')
        ]
        graph_ref = build_graph(nodes_attributes, edges)
        (flag, resp) = compare_graphs(graph, graph_ref, 'some_op', check_op_attrs=True)
        self.assertTrue(flag, resp)
