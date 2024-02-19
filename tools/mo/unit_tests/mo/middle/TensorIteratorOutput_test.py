# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.middle.TensorIteratorOutput import SmartOutputMatcher
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph_with_attrs


class SmartOutputMatcherTests(unittest.TestCase):
    def test(self):
        pattern_matcher = SmartOutputMatcher()
        pattern = pattern_matcher.pattern()

        graph = build_graph_with_attrs(nodes_with_attrs=pattern['nodes'], edges_with_attrs=pattern['edges'],
                                       # update_edge_attrs=None,
                                        new_nodes_with_attrs=[('index', {'kind': 'data'}),
                                                              ('value', {'kind': 'data'}),
                                                              ('ta_size', {'kind': 'data'}),
                                                              ],
                                        new_edges_with_attrs=[('index', 'TensorArrayWrite', {'in':1}),
                                                              ('value', 'TensorArrayWrite', {'in': 2}),
                                                              ('ta_size', 'TensorArray')
                                                              ],
                                       update_nodes_attributes=[('WriteEnter_data', {'value': np.array([1, 1])}),

                                                                ('start_data', {'value': np.array([0])}),
                                                                ('delta_data', {'value': np.array([1])}),
                                                                ])

        pattern_matcher.find_and_replace_pattern(graph)

        graph_ref = build_graph_with_attrs(
            nodes_with_attrs=[
                              ('TensorIteratorOutput', {'kind': 'op', 'op': 'TensorIteratorOutput'}),
                              ('TensorArrayGather_data', {'kind': 'data'}),
                              ('index', {'kind': 'data'}),
                              ('value', {'kind': 'data'}),
                              ('ta_size', {'kind': 'data'}), ],
            edges_with_attrs=[('ta_size', 'TensorIteratorOutput', {'in': 0}),
                              ('index', 'TensorIteratorOutput', {'in': 2}),
                              ('value', 'TensorIteratorOutput', {'in': 1}),
                              ('TensorIteratorOutput', 'TensorArrayGather_data')],
            update_edge_attrs=None,
            new_nodes_with_attrs=[],
            new_edges_with_attrs=[],
            )
        (flag, resp) = compare_graphs(graph, graph_ref, 'TensorArrayGather_data', check_op_attrs=True)
        self.assertTrue(flag, resp)
