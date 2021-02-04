"""
 Copyright (C) 2018-2021 Intel Corporation

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

from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph, regular_op

nodes = {
    **regular_op('input', {'type': 'Parameter'}),
    **regular_op('Op1', {'type': 'Op1', 'kind': 'op', 'op': 'Op1'}),
    **regular_op('Op2', {'type': 'Op2', 'kind': 'op', 'op': 'Op2'}),
    **regular_op('Op3', {'type': 'Op3', 'kind': 'op', 'op': 'Op3'}),

    'input_data': {'kind': 'data', 'fw_tensor_debug_info': [('input', 0, 'input'), ('Op1', 0, 'Op1,Op2')]},
    'Op1_data': {'kind': 'data', 'fw_tensor_debug_info': [('Op1', 0, 'Op1,Op2')]},
    'Op2_data': {'kind': 'data'},
    'Op3_data': {'kind': 'data', 'fw_tensor_debug_info': [('Op3', 0, 'Op3')]},
}


class TestsGetTensorNames(unittest.TestCase):
    def test_front(self):
        graph = build_graph(nodes,
                            [('input', 'Op1', {'in': 0, 'out': 0, 'fw_tensor_debug_info': [('input', 0, 'input'),
                                                                                           ('Op1', 0, 'Op1,Op2')]})])
        graph.stage = 'front'
        input_node = Node(graph, 'input')
        self.assertTrue(input_node.out_port(0).get_tensor_names() == 'input,Op1\\,Op2')

        op1_node = Node(graph, 'Op1')
        op1_node.add_output_port(0)
        self.assertTrue(op1_node.out_port(0).get_tensor_names() is None)

    def test_middle(self):
        graph = build_graph(nodes, [('input', 'input_data'), ('input_data', 'Op1'),
                                    ('input_data', 'Op2')])

        input_node = Node(graph, 'input')
        self.assertTrue(input_node.out_port(0).get_tensor_names() == 'input,Op1\\,Op2')

        op1_node = Node(graph, 'Op1')
        op1_node.add_output_port(0)
        self.assertTrue(op1_node.out_port(0).get_tensor_names() is None)

        op2_node = Node(graph, 'Op2')
        op2_node.add_output_port(0)
        self.assertTrue(op2_node.out_port(0).get_tensor_names() is None)

    def test_port_renumber(self):
        graph = build_graph(nodes, [('input', 'input_data'), ('input_data', 'Op1'),
                                    ('Op1', 'Op1_data', {'out': 1}), ('Op1_data', 'Op2')])
        input_node = Node(graph, 'input')
        self.assertTrue(input_node.out_port(0).get_tensor_names(port_renumber=True) == 'input,Op1\\,Op2')

        op1_node = Node(graph, 'Op1')
        op1_node.add_output_port(0)

        self.assertTrue(op1_node.out_port(0).get_tensor_names(port_renumber=True) == 'Op1\\,Op2')

    def test_reconnect_middle_case1(self):
        graph = build_graph(nodes, [('input', 'input_data'), ('input_data', 'Op1'), ('Op3', 'Op3_data')])
        input_node = Node(graph, 'input')

        input_node_out_port = input_node.out_port(0)
        self.assertTrue(input_node_out_port.get_tensor_names() == 'input,Op1\\,Op2')

        op3_node = Node(graph, 'Op3')
        input_node_out_port.get_connection().set_source(op3_node.out_port(0))

        self.assertTrue(input_node_out_port.get_tensor_names() is None)
        self.assertTrue(op3_node.out_port(0).get_tensor_names() == 'Op3,input,Op1\\,Op2')

    def test_reconnect_front_case1(self):
        graph = build_graph(nodes, [('input', 'Op1', {'in': 0, 'out': 0, 'fw_tensor_debug_info': [('input', 0, 'input'),
                                                                                                  ('Op1', 0,
                                                                                                   'Op1,Op2')]}),
                                    ('Op3', 'Op2', {'in': 0, 'out': 0, 'fw_tensor_debug_info': [('Op3', 0, 'Op3')]})])
        graph.stage = 'front'
        input_node = Node(graph, 'input')

        input_node_out_port = input_node.out_port(0)
        self.assertTrue(input_node_out_port.get_tensor_names() == 'input,Op1\\,Op2')

        op3_node = Node(graph, 'Op3')
        input_node_out_port.get_connection().set_source(op3_node.out_port(0))

        self.assertTrue(input_node_out_port.get_tensor_names() is None)
        self.assertTrue(op3_node.out_port(0).get_tensor_names() == 'Op3,input,Op1\\,Op2')

    def test_reconnect_middle_case1(self):
        graph = build_graph(nodes, [('input', 'input_data'), ('input_data', 'Op1'), ('Op3', 'Op3_data')])
        input_node = Node(graph, 'input')

        input_node_out_port = input_node.out_port(0)
        self.assertTrue(input_node_out_port.get_tensor_names() == 'input,Op1\\,Op2')

        op3_node = Node(graph, 'Op3')
        input_node_out_port.get_connection().set_source(op3_node.out_port(0))

        self.assertTrue(input_node_out_port.get_tensor_names() is None)
        self.assertTrue(op3_node.out_port(0).get_tensor_names() == 'Op3,input,Op1\\,Op2')

    def test_reconnect_middle_case2(self):
        graph = build_graph(nodes, [('input', 'input_data'), ('input_data', 'Op1', {'out': 0}),
                                    ('input_data', 'Op1', {'out': 1}), ('Op3', 'Op3_data')])
        input_node = Node(graph, 'input')

        input_node_out_port = input_node.out_port(0)
        self.assertTrue(input_node_out_port.get_tensor_names() == 'input,Op1\\,Op2')

        op3_node = Node(graph, 'Op3')
        input_node_out_port.get_connection().set_source(op3_node.out_port(0))

        self.assertTrue(input_node_out_port.get_tensor_names() is None)
        self.assertTrue(op3_node.out_port(0).get_tensor_names() == 'Op3,input,Op1\\,Op2')
