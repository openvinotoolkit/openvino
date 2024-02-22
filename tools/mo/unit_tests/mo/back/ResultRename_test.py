# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.back.ResultRename import ResultRename
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, result

nodes = {
    'Op1': {'type': 'Op1', 'kind': 'op', 'op': 'Op1'},
    'Op2': {'type': 'Op2', 'kind': 'op', 'op': 'Op2'},
    'Op1_data': {'kind': 'data', 'fw_tensor_debug_info': [('Op1', 'Op1_tensor')]},
    'Op2_data': {'kind': 'data', 'fw_tensor_debug_info': [('Op2', 'Op2_tensor')]},
    **result('result1'),
    **result('result2'),
}


class ResultRenameTest(unittest.TestCase):
    def test_case1(self):
        graph = build_graph(nodes, [('Op1', 'Op1_data'), ('Op1_data', 'result1')])
        ResultRename().find_and_replace_pattern(graph)
        res_node = Node(graph, 'result1')
        self.assertTrue(res_node['name'] == 'Op1_tensor')

    def test_case2(self):
        graph = build_graph(nodes, [])
        graph_ref = build_graph(nodes, [])

        ResultRename().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, 'result1', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_case3(self):
        graph = build_graph(nodes, [('Op1', 'Op1_data'), ('Op1_data', 'result1')])
        res_node_graph = Node(graph, 'Op1')
        res_node_graph['name'] = 'Op1_tensor'
        ResultRename().find_and_replace_pattern(graph)
        res_node = Node(graph, 'result1')
        self.assertTrue(res_node['name'] == 'Op1_tensor/sink_port_0')

    def test_case4(self):
        graph = build_graph(nodes, [('Op1', 'Op1_data'), ('Op1_data', 'result1'),
                                    ('Op1_data', 'Op2'), ('Op2', 'Op2_data'),
                                    ('Op2_data', 'result2')])
        graph.outputs_order = ['result1', 'result2']

        ResultRename().find_and_replace_pattern(graph)
        res1_node = Node(graph, 'result1')
        res2_node = Node(graph, 'result2')
        self.assertTrue(res1_node['name'] == 'Op1_tensor')
        self.assertTrue(res2_node['name'] == 'Op2_tensor')

        self.assertTrue(graph.outputs_order == ['Op1_tensor', 'Op2_tensor'])

    def test_case5(self):
        graph = build_graph(nodes, [('Op1', 'Op1_data'), ('Op1_data', 'result1'),
                                    ('Op1_data', 'Op2'), ('Op2', 'Op2_data'),
                                    ('Op2_data', 'result2')])

        res_node_graph = Node(graph, 'result1')
        res_node_graph['name'] = 'Op1_tensor'
        ResultRename().find_and_replace_pattern(graph)
        res1_node = Node(graph, 'result1')
        res2_node = Node(graph, 'result2')
        self.assertTrue(res1_node['name'] == 'Op1_tensor')
        self.assertTrue(res2_node['name'] == 'Op2_tensor')

    def test_case6(self):
        _nodes = nodes.copy()
        _nodes.update({
            'Op3': {'type': 'Op3', 'kind': 'op', 'op': 'Op3'},
            'Op4': {'type': 'Op4', 'kind': 'op', 'op': 'Op4'},
            'Op3_data': {'kind': 'data', 'fw_tensor_debug_info': [('Op3', 'Op3_tensor')]},
            'Op4_data': {'kind': 'data', 'fw_tensor_debug_info': [('Op4', 'Op4_tensor')]},
            **result('result3'),
            **result('result4'),
        })
        graph = build_graph(_nodes, [('Op1', 'Op1_data'), ('Op1_data', 'result1'), ('Op1_data', 'Op2'),
                                     ('Op2', 'Op2_data'), ('Op2_data', 'result2'), ('Op2_data', 'Op3'),
                                     ('Op3', 'Op3_data'), ('Op3_data', 'result3'), ('Op3_data', 'Op4'),
                                     ('Op4', 'Op4_data'), ('Op4_data', 'result4')])
        graph.outputs_order = ['result1', 'result3', 'result4', 'result2']

        ResultRename().find_and_replace_pattern(graph)
        self.assertTrue(Node(graph, 'result1')['name'] == 'Op1_tensor')
        self.assertTrue(Node(graph, 'result2')['name'] == 'Op2_tensor')
        self.assertTrue(Node(graph, 'result3')['name'] == 'Op3_tensor')
        self.assertTrue(Node(graph, 'result4')['name'] == 'Op4_tensor')

        self.assertTrue(graph.outputs_order == ['Op1_tensor', 'Op3_tensor', 'Op4_tensor', 'Op2_tensor'])
