# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from extensions.back.PadNormalizer import PadNormalizer
from mo.front.common.partial_infer.utils import int64_array, float32_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, result, regular_op_with_empty_data, connect, valued_const_with_data


class PadNormalizerTest(unittest.TestCase):
    def test_1(self):
        nodes = {
            **regular_op_with_empty_data('input', {'type': 'Parameter', 'shape': int64_array([1, 3, 768, 1280])}),
            **valued_const_with_data('add_const', float32_array([-127.5, -127.5, -127.5])),
            **regular_op_with_empty_data('add', {'type': 'Add', 'kind': 'op', 'op': 'Add'}),
            **valued_const_with_data('pad_const_1', int64_array([0, 0, 0, 0])),
            **valued_const_with_data('pad_const_2', int64_array([0, 0, 1, 1])),
            **regular_op_with_empty_data('pad', {'type': 'Pad', 'kind': 'op', 'op': 'Pad', 'mode': 'constant'}),
            **valued_const_with_data('convert_like_const', float32_array(0.0)),
            **regular_op_with_empty_data('convert_like', {'type': 'ConvertLike', 'kind': 'op', 'op': 'ConvertLike'}),
            **result('result'),
        }
        edges = [*connect('input', '0:add'),
                 *connect('add_const', '1:add'),
                 *connect('add', '0:pad'),
                 *connect('pad_const_1', '1:pad'),
                 *connect('pad_const_2', '2:pad'),
                 *connect('pad', 'result'),
                 ]

        ref_edges = [*connect('input', '0:add'),
                     *connect('add_const', '1:add'),
                     *connect('add', '0:pad'),
                     *connect('convert_like_const', '0:convert_like'),
                     ('add_d', 'convert_like'),
                     *connect('pad_const_1', '1:pad'),
                     *connect('pad_const_2', '2:pad'),
                     *connect('convert_like', '3:pad'),
                     *connect('pad', 'result'),
                 ]

        graph = build_graph(nodes, edges)
        graph.stage = 'back'
        PadNormalizer().find_and_replace_pattern(graph)

        graph_ref = build_graph(nodes, ref_edges)
        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)
