# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.front.OneHotDepthNormalizer import OneHotDepthNormalizer
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, result, \
    regular_op, const


class OneHotDepthNormalizerTest(unittest.TestCase):
    def test(self):
        nodes = {
            **regular_op('input', {'type': 'Parameter'}),
            **const('depth', int64_array([2])),
            **regular_op('onehot', {'type': 'OneHot', 'kind': 'op', 'op': 'OneHot'}),

            **regular_op('reshape', {'type': 'Reshape', 'kind': 'op', 'op': 'Reshape'}),
            **const('reshape_dims', int64_array([])),
            **result('result'),
        }
        edges = [('input', 'onehot'),
                 ('depth', 'onehot'),
                 ('onehot', 'result'),
                 ]
        graph = build_graph(nodes, edges)

        graph.graph['layout'] = 'NCHW'
        graph.stage = 'front'

        edges_ref = [('input', 'onehot'),
                     ('depth', 'reshape'),
                     ('reshape_dims', 'reshape'),
                     ('reshape', 'onehot'),
                     ('onehot', 'result'),
                     ]

        graph_ref = build_graph(nodes, edges_ref)

        OneHotDepthNormalizer().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result')
        self.assertTrue(flag, resp)
