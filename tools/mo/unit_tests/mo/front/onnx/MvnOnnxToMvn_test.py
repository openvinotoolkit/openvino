# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.front.onnx.MvnOnnxToMvn import MvnOnnxToMvn
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, regular_op_with_empty_data, result, const, connect_front

nodes = {
    **regular_op_with_empty_data('input', {'type': 'Parameter'}),
    **regular_op_with_empty_data('mvn_onnx', {'op': 'MVNOnnx',
                                              'axes': int64_array([2, 3]),
                                              'eps': 1e-9,
                                              'eps_mode': 'outside_sqrt',
                                              'normalize_variance': 1}),
    **result(),

    # nodes after replacement
    **const('axes', int64_array([2, 3])),
    **regular_op_with_empty_data('mvn', {'op': 'MVN', 'type': None}),
}


class MvnOnnxToMvnTest(unittest.TestCase):
    def test_mvn_normalize(self):
        graph = build_graph(nodes, [('input', 'mvn_onnx'),
                                    ('mvn_onnx', 'output')],
                            nodes_with_edges_only=True)
        graph.stage = 'front'

        MvnOnnxToMvn().find_and_replace_pattern(graph)

        graph_ref = build_graph(nodes, [('input', 'mvn'),
                                        *connect_front('axes', '1:mvn'),
                                        ('mvn', 'output')],
                                nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)
