# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from extensions.front.Softplus_fusion import SoftplusFusion
from mo.front.common.partial_infer.utils import float_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph, const, regular_op, result, build_graph_with_edge_attrs

ref_nodes = {**regular_op('input', {'type': 'Parameter'}),
             **regular_op('softplus', {'type': 'SoftPlus', 'name': 'final_log'}),
             **result('result')
             }
ref_edges = [('input', 'softplus'), ('softplus', 'result')]


class SoftplusFusionTest(unittest.TestCase):
    nodes = {
        **regular_op('input', {'type': 'Parameter'}),
        **regular_op('exp', {'op': 'Exp'}),
        **const('const_1', float_array([1.0])),
        **regular_op('add', {'op': 'Add'}),
        **regular_op('ln', {'op': 'Log', 'name': 'final_log'}),
        **result('result'),
    }

    edges = [('input', 'exp', {'in': 0, 'out': 0}),
             ('const_1', 'add', {'in': 0, 'out': 0}),
             ('exp', 'add', {'in': 1, 'out': 0}),
             ('add', 'ln', {'in': 0, 'out': 0}),
             ('ln', 'result', {'in': 0, 'out': 0})]

    def test_softplus_fusion_test(self):
        graph = build_graph_with_edge_attrs(self.nodes, self.edges, {})

        graph_ref = build_graph(ref_nodes, ref_edges)
        graph.stage = 'front'

        SoftplusFusion().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result')
        self.assertTrue(flag, resp)
        self.assertTrue(len(graph.get_op_nodes(name='final_log')) == 1 and
                        graph.get_op_nodes(name='final_log')[0].op == 'SoftPlus')

    def test_softplus_fusion_test_wrong_const(self):
        graph = build_graph_with_edge_attrs(self.nodes, self.edges, {'const_1': {'value': float_array([0.9999])}})

        graph_ref = graph.copy()
        graph.stage = 'front'

        SoftplusFusion().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result')
        self.assertTrue(flag, resp)

