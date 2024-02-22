# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.front.reduce_axis_normalizer import ReduceAxisNormalizer
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, result, connect_front, regular_op

nodes = {
    **regular_op('parameter', {'type': 'Parameter'}),
    **regular_op('reduce', {'op': 'ReduceSum', 'axis': None}),
    **regular_op('axis', {'op': 'Const', 'type': 'Const', 'value': int64_array([1])}),
    **result(),
}

edges = [
    *connect_front('parameter:0', '0:reduce'),
    *connect_front('reduce', 'output'),
]


class ReduceAxisNormalizerTest(unittest.TestCase):
    def test_reduce_axis_is_None(self):
        graph = build_graph(nodes, edges, nodes_with_edges_only=True)
        graph.stage = 'front'

        ReduceAxisNormalizer().find_and_replace_pattern(graph)

        ref_nodes = nodes.copy()
        ref_nodes.update({**regular_op('rank', {'op': 'Rank', 'type': None}),
                          **regular_op('range', {'op': 'Range', 'type': 'Range'}),
                          **regular_op('begin', {'type': 'Const', 'value': int64_array([0])}),
                          **regular_op('step', {'type': 'Const', 'value': int64_array([1])}),
                          })
        graph_ref = build_graph(ref_nodes, [
            *edges,
            *connect_front('parameter:0', 'rank'),
            *connect_front('begin:0', '0:range'),
            *connect_front('rank:0', '1:range'),
            *connect_front('step:0', '2:range'),
            *connect_front('range:0', '1:reduce'),
        ], nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_reduce_axis_is_const(self):
        graph = build_graph(nodes, edges, {'reduce': {'axis': 1}}, nodes_with_edges_only=True)
        graph.stage = 'front'

        graph_ref = build_graph(nodes, [
            *edges,
            *connect_front('axis', '1:reduce'),
        ], {'axis': {'value': np.int64(1)}}, nodes_with_edges_only=True)

        ReduceAxisNormalizer().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)
