# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from extensions.front.kaldi.set_ports import SetPortsPattern
from mo.utils.error import Error
from mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, regular_op, connect_front, Node


class SetPortsTest(unittest.TestCase):
    def test_set_ports_chain(self):
        nodes = {
            **regular_op('op1', {}),
            **regular_op('op2', {}),
            **regular_op('op3', {}),
        }

        graph = build_graph(nodes, [
            ('op1', 'op2', {'fw_tensor_debug_info': {}}),
            ('op2', 'op3', {'fw_tensor_debug_info': {}})
        ], nodes_with_edges_only=True)

        graph.stage = 'front'

        ref_graph = build_graph(nodes, [
            *connect_front('op1:0', '0:op2'),
            *connect_front('op2:0', '0:op3')
        ], nodes_with_edges_only=True)

        SetPortsPattern().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, ref_graph, 'op3', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_set_ports_split(self):
        nodes = {
                **regular_op('op1', {}),
                **regular_op('split', {'op': 'Split'}),
                **regular_op('op2', {}),
                **regular_op('op3', {}),
                **regular_op('op4', {}),
            }

        graph = build_graph(nodes, [
            ('op1', 'split', {'fw_tensor_debug_info': {}}),
            ('split', 'op2', {'fw_tensor_debug_info': {}, 'out_port': 0}),
            ('split', 'op3', {'fw_tensor_debug_info': {}, 'out_port': 1}),
            ('split', 'op4', {'fw_tensor_debug_info': {}, 'out_port': 2})
        ], nodes_with_edges_only=True)

        graph.stage = 'front'
        graph.nodes()['split']['out_ports_count'] = 3

        ref_graph = build_graph(nodes, [
            *connect_front('op1:0', '0:split'),
            *connect_front('split:0', '0:op2'),
            *connect_front('split:1', '0:op3'),
            *connect_front('split:2', '0:op4')
        ], nodes_with_edges_only=True)

        SetPortsPattern().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, ref_graph, 'op4', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_set_ports_split2(self):
        nodes = {
            **regular_op('op1', {}),
            **regular_op('split', {'op': 'Split'}),
            **regular_op('op2', {}),
            **regular_op('op3', {}),
            **regular_op('op4', {}),
        }

        graph = build_graph(nodes, [
            ('op1', 'split', {'fw_tensor_debug_info': {}}),
            ('split', 'op2', {'fw_tensor_debug_info': {}, 'out_port': 0}),
            ('split', 'op4', {'fw_tensor_debug_info': {}, 'out_port': 4}),
            ('split', 'op3', {'fw_tensor_debug_info': {}, 'out_port': 6})
        ], nodes_with_edges_only=True)

        graph.stage = 'front'
        graph.nodes()['split']['out_ports_count'] = 3

        ref_graph = build_graph(nodes, [
            *connect_front('op1:0', '0:split'),
            *connect_front('split:0', '0:op2'),
            *connect_front('split:1', '0:op4'),
            *connect_front('split:2', '0:op3')
        ], nodes_with_edges_only=True)

        SetPortsPattern().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, ref_graph, 'op4', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_set_ports_split3(self):
        nodes = {
            **regular_op('op1', {}),
            **regular_op('split', {'op': 'Split'}),
            **regular_op('op2', {}),
            **regular_op('op3', {}),
            **regular_op('op4', {}),
        }

        graph = build_graph(nodes, [
            ('op1', 'split',  {'fw_tensor_debug_info': {}}),
            ('split', 'op2',  {'fw_tensor_debug_info': {}, 'out_port': 0}),
            ('split', 'op3',  {'fw_tensor_debug_info': {}, 'out_port': 1}),
            ('split', 'op4',  {'fw_tensor_debug_info': {}, 'out_port': 2})
        ], nodes_with_edges_only=True)

        Node(graph, 'split').out_port(0).get_connection().add_destination(Node(graph, 'op4').in_port(0))
        graph.nodes()['split']['out_ports_count'] = 2

        graph.stage = 'front'

        self.assertRaises(Error, SetPortsPattern().find_and_replace_pattern, graph)
