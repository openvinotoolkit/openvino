# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.middle.TensorIteratorBackEdge import BackEdgesMatching
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph_with_attrs


class BackEdgesMatchingTests(unittest.TestCase):
    def test_no_exit(self):
        pattern_matcher = BackEdgesMatching()
        pattern = pattern_matcher.pattern()
        graph = build_graph_with_attrs(nodes_with_attrs=pattern['nodes'], edges_with_attrs=pattern['edges'], update_edge_attrs=None,
                                       new_nodes_with_attrs=[('from_body_data', {'kind':'data'})],
                                       new_edges_with_attrs=[('from_body_data', 'NextIteration')])

        pattern_matcher.find_and_replace_pattern(graph)
        graph_ref = build_graph_with_attrs(nodes_with_attrs=[('condition', {'kind': 'op', 'op':'TensorIteratorCondition'}),
                                                      ('condition_data', {'kind': 'data'}),
                                                      ('back_edge', {'kind': 'op', 'op': 'TensorIteratorBackEdge'}),
                                                      ('enter_data', {'kind': 'data'}),
                                                      ('from_body_data', {'kind': 'data'}),
                                                      ('Identity_1_data', {'kind': 'data'}),],
                                    edges_with_attrs=[('condition', 'condition_data'),
                                           ('enter_data', 'back_edge', {'in': 0}),
                                           ('condition_data', 'back_edge', {'in': 2}),  # {in:2}
                                           ('from_body_data', 'back_edge', {'in': 1}),
                                           ('back_edge', 'Identity_1_data')],
                                    update_edge_attrs=None,
                                    new_nodes_with_attrs=[],
                                    new_edges_with_attrs=[],
                                )
        (flag, resp) = compare_graphs(graph, graph_ref, 'Identity_1_data', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_with_exit(self):
        pattern_matcher = BackEdgesMatching()
        pattern = pattern_matcher.pattern()
        graph = build_graph_with_attrs(nodes_with_attrs=pattern['nodes'], edges_with_attrs=pattern['edges'], update_edge_attrs=None,
                                new_nodes_with_attrs=[('from_body_data', {'kind': 'data'}),
                                           ('exit', {'kind': 'op', 'op': 'Exit', 'name': 'exit'}),
                                           ('exit_data', {'kind': 'data'}),
                                           ('Switch_1_data_exit', {'kind': 'data'})],

                                new_edges_with_attrs=[('from_body_data', 'NextIteration'),
                                           ('Switch_1', 'Switch_1_data_exit', {'out': 0}),
                                           ('Switch_1_data_exit', 'exit', {'out': 0}),
                                           ('exit', 'exit_data')])

        pattern_matcher.find_and_replace_pattern(graph)
        graph_ref = build_graph_with_attrs(nodes_with_attrs=[('condition', {'kind': 'op', 'op':'TensorIteratorCondition'}),
                                                      ('condition_data', {'kind': 'data'}),
                                                      ('back_edge', {'kind': 'op', 'op': 'TensorIteratorBackEdge'}),
                                                      ('enter_data', {'kind': 'data'}),
                                                      ('from_body_data', {'kind': 'data'}),
                                                      ('Identity_1_data', {'kind': 'data'}),
                                                      ('output', {'kind':'op', 'op':'TensorIteratorOutput'}),
                                                      ('exit_data', {'kind': 'data'})
                                                        ],
                                            edges_with_attrs=[('condition', 'condition_data'),
                                                   ('enter_data', 'back_edge', {'in': 0}),
                                                   ('condition_data', 'back_edge', {'in': 2}),
                                                   ('from_body_data', 'back_edge', {'in': 1}),
                                                   ('back_edge', 'Identity_1_data'),
                                                   ('condition_data', 'output'),
                                                   ('output', 'exit_data'),
                                                   ('from_body_data', 'output')],
                                            update_edge_attrs=None,
                                            new_nodes_with_attrs=[],
                                            new_edges_with_attrs=[],
                                )
        (flag, resp) = compare_graphs(graph, graph_ref, 'Identity_1_data', check_op_attrs=True)
        self.assertTrue(flag, resp)
