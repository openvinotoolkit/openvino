# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from extensions.back.ResultRename import ResultRename
from mo.graph.graph import Node
from mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, regular_op, result

nodes = {
    **regular_op('Op1', {'type': 'Op1', 'kind': 'op', 'op': 'Op1'}),
    **regular_op('Op2', {'type': 'Op2', 'kind': 'op', 'op': 'Op2'}),
    **result('result1'),
    **result('result2'),
    'Op1_data': {'kind': 'data', 'fw_tensor_debug_info': [('Op1', 'Op1_tensor')]},
    'Op2_data': {'kind': 'data', 'fw_tensor_debug_info': [('Op1', 'Op2_tensor')]},
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
        ResultRename().find_and_replace_pattern(graph)
        res1_node = Node(graph, 'result1')
        res2_node = Node(graph, 'result2')
        self.assertTrue(res1_node['name'] == 'Op1_tensor')
        self.assertTrue(res2_node['name'] == 'Op2_tensor')

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

