# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import Mock

from openvino.tools.mo.ops.assert_op import Assert
from openvino.tools.mo.graph.graph import Node
from unit_tests.utils.graph import build_graph_with_edge_attrs


class TestAssert(unittest.TestCase):
    def test_assert_cf_true(self):
        me_mock = Mock()
        nodes = {
            'input_data': {'kind': 'data', 'executable': True},
            'assert': {'type': 'Assert', 'value': None, 'kind': 'op', 'op': 'Assert'},
            'assert_data': {'value': True, 'kind': 'data', 'executable': True}}
        edges = [
            ('input_data', 'assert', {'in': 0}),
            ('assert', 'assert_data', {'out': 0, 'control_flow_edge': False})]
        graph = build_graph_with_edge_attrs(nodes, edges)
        tested_class = Assert(graph=graph, attrs={})
        node = Node(graph, 'assert')
        tested_class.assert_control_flow_infer(node=node, is_executable=True, mark_executability=me_mock)
        me_mock.assert_called_once_with('assert_data', True)

    def test_assert_cf_false(self):
        me_mock = Mock()
        nodes = {
            'input_data': {'name': 'input', 'kind': 'data', 'executable': True},
            'assert': {'name': 'assert', 'type': 'Assert', 'value': None, 'kind': 'op', 'op': 'Assert'},
            'assert_data': {'name': 'output', 'value': False, 'kind': 'data', 'executable': True}}
        edges = [
            ('input_data', 'assert', {'in': 0}),
            ('assert', 'assert_data', {'out': 0, 'control_flow_edge': False})]
        graph = build_graph_with_edge_attrs(nodes, edges)
        tested_class = Assert(graph=graph, attrs={})
        node = Node(graph, 'assert')
        tested_class.assert_control_flow_infer(node=node, is_executable=True, mark_executability=me_mock)
        me_mock.assert_called_once_with('assert_data', False)
