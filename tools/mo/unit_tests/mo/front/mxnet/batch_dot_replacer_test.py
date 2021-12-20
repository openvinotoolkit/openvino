# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.front.mxnet.batch_dot_replacer import BatchDotReplacer
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, regular_op, result, connect_front


class BatchDotReplacerTest(unittest.TestCase):

    def test_1(self):
        graph = build_graph(
            nodes_attrs={
                **regular_op('input', {'op': 'Parameter', 'type': 'Parameter'}),
                **regular_op('const_input', {'op': 'Const', 'type': 'Const'}),
                **regular_op('batch_dot', {'op': 'batch_dot', 'type': None, 'transpose_a': True, 'transpose_b': False}),
                **result('result')
            },
            edges=[
                *connect_front('input', '0:batch_dot'),
                *connect_front('const_input', '1:batch_dot'),
                *connect_front('batch_dot', 'result'),
            ]
        )

        ref_graph = build_graph(
            nodes_attrs={
                **regular_op('input', {'op': 'Parameter', 'type': 'Parameter'}),
                **regular_op('const_input', {'op': 'Const', 'type': 'Const'}),
                **regular_op('mat_mul', {'op': 'MatMul', 'type': 'MatMul', 'transpose_a': True, 'transpose_b': False}),
                **result('result')
            },
            edges=[
                *connect_front('input', '0:mat_mul'),
                *connect_front('const_input', '1:mat_mul'),
                *connect_front('mat_mul', 'result'),
            ]
        )
        graph.stage = 'front'
        BatchDotReplacer().find_and_replace_pattern(graph)
        flag, resp = compare_graphs(graph, ref_graph, 'result', 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)
