# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.ops.lstm_cell import LSTMCell
from openvino.tools.mo.graph.graph import Node
from unit_tests.utils.graph import build_graph, regular_op

nodes = {
    **regular_op('Op1', {'type': 'Op1', 'kind': 'op', 'op': 'Op1'}),
    **regular_op('Op2', {'type': 'Op2', 'kind': 'op', 'op': 'Op2'}),
    **regular_op('Op3', {'type': 'Op3', 'kind': 'op', 'op': 'Op3'}),
}


class TestOp(unittest.TestCase):
    def test_create_node(self):
        graph = build_graph(nodes, [('Op1', 'Op3', {'in': 0, 'out': 0, 'fw_tensor_debug_info': [('Op1', 'Op1')]}),
                                    ('Op2', 'Op3', {'in': 1, 'out': 0, 'fw_tensor_debug_info': [('Op2', 'Op2')]})])
        graph.stage = 'front'
        input1 = Node(graph, 'Op1')
        input2 = Node(graph, 'Op2')
        inputs = [(input1, 0), (input2, 0)]

        lstm_op = LSTMCell(graph, dict(name='LSTMCell'))
        _ = lstm_op.create_node(inputs)

        self.assertTrue(input1.out_edge(0)['fw_tensor_debug_info'] == [('Op1', 'Op1')])
        self.assertTrue(input2.out_edge(0)['fw_tensor_debug_info'] == [('Op2', 'Op2')])
