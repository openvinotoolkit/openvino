"""
 Copyright (C) 2018-2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import unittest
from unittest.mock import Mock

from extensions.ops.assert_op import Assert
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph_with_edge_attrs


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
