# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.front.output_cut import OutputCut
from openvino.tools.mo.graph.graph import Node
from unit_tests.utils.graph import build_graph, regular_op

nodes = {
    **regular_op('Parameter1', {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'}),
    **regular_op('Op1', {'type': 'Op1', 'kind': 'op', 'op': 'Op1'}),
    **regular_op('Op2', {'type': 'Op2', 'kind': 'op', 'op': 'Op2'}),

    **regular_op('FakeOutput1', {'type': 'Identity', 'kind': 'op', 'op': 'Identity', 'needs_removal': True, 'id': 0}),
    **regular_op('FakeOutput2', {'type': 'Identity', 'kind': 'op', 'op': 'Identity', 'needs_removal': True, 'id': 1}),

}


class TestsOutputCut(unittest.TestCase):
    def test_case1(self):
        graph = build_graph(nodes, [('Parameter1', 'FakeOutput1',
                                     {'in': 0, 'out': 0, 'fw_tensor_debug_info':
                                         [('Parameter1', 'Parameter1_tensor_name')]})])
        graph.graph['packed_outputs'] = None
        graph.graph['user_shapes'] = None

        graph.stage = 'front'
        OutputCut().find_and_replace_pattern(graph)

        param1 = Node(graph, 'Parameter1')
        self.assertTrue(param1.out_node()['type'] == 'Result')
        self.assertTrue(param1.out_edge()['fw_tensor_debug_info'] == [('Parameter1', 'Parameter1_tensor_name')])
        self.assertTrue(graph.get_op_nodes(name='FakeOutput1') == [])

    def test_case2(self):
        graph = build_graph(nodes, [('Parameter1', 'Op1'),
                                    ('Op1', 'FakeOutput1',
                                     {'in': 1, 'out': 1, 'fw_tensor_debug_info':
                                         [('Op1', 'Op1_tensor_name')]}),
                                    ('Parameter1', 'Op2'),
                                    ('Op2', 'FakeOutput2',
                                     {'in': 2, 'out': 3,
                                      'fw_tensor_debug_info': [('Op2', 'Op2_tensor_name')]})])
        graph.graph['packed_outputs'] = None
        graph.graph['user_shapes'] = None

        graph.stage = 'front'
        OutputCut().find_and_replace_pattern(graph)

        op1 = Node(graph, 'Op1')
        op2 = Node(graph, 'Op2')
        self.assertTrue(op1.out_node(1)['type'] == 'Result')
        self.assertTrue(op2.out_node(3)['type'] == 'Result')
        self.assertTrue(op1.out_edge(1)['fw_tensor_debug_info'] == [('Op1', 'Op1_tensor_name')])
        self.assertTrue(op2.out_edge(3)['fw_tensor_debug_info'] == [('Op2', 'Op2_tensor_name')])
        self.assertTrue(graph.get_op_nodes(name='FakeOutput1') == [])
        self.assertTrue(graph.get_op_nodes(name='FakeOutput2') == [])

    def test_case3(self):
        graph = build_graph(nodes, [])
        graph.graph['packed_outputs'] = None
        graph.graph['user_shapes'] = None

        graph.stage = 'front'
        OutputCut().find_and_replace_pattern(graph)

        self.assertTrue(graph.get_op_nodes(name='FakeOutput1') == [])
        self.assertTrue(graph.get_op_nodes(name='FakeOutput2') == [])
