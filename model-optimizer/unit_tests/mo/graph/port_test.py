# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from mo.graph.graph import Node
from unit_tests.utils.graph import build_graph, regular_op, valued_const_with_data, result, connect

nodes = {
    **regular_op('input', {'type': 'Parameter'}),
    **regular_op('Op1', {'type': 'Op1', 'kind': 'op', 'op': 'Op1'}),
    **regular_op('Op2', {'type': 'Op2', 'kind': 'op', 'op': 'Op2'}),
    **regular_op('Op3', {'type': 'Op3', 'kind': 'op', 'op': 'Op3'}),

    'input_data': {'kind': 'data', 'fw_tensor_debug_info': [('input', 'input'), ('Op1', 'Op1,Op2')]},
    'Op1_data': {'kind': 'data', 'fw_tensor_debug_info': [('Op1', 'Op1,Op2')]},
    'Op2_data': {'kind': 'data'},
    'Op3_data': {'kind': 'data', 'fw_tensor_debug_info': [('Op3', 'Op3')]},
}


class TestsGetTensorNames(unittest.TestCase):
    def test_front(self):
        graph = build_graph(nodes,
                            [('input', 'Op1', {'in': 0, 'out': 0, 'fw_tensor_debug_info': [('input', 'input'),
                                                                                           ('Op1', 'Op1,Op2')]})])
        graph.stage = 'front'
        input_node = Node(graph, 'input')
        self.assertTrue(input_node.out_port(0).get_tensor_names() == ['Op1\\,Op2', 'input'])

        op1_node = Node(graph, 'Op1')
        op1_node.add_output_port(0)
        self.assertTrue(op1_node.out_port(0).get_tensor_names() == [])

    def test_middle(self):
        graph = build_graph(nodes, [('input', 'input_data'), ('input_data', 'Op1'),
                                    ('input_data', 'Op2')])

        input_node = Node(graph, 'input')
        self.assertTrue(input_node.out_port(0).get_tensor_names() == ['Op1\\,Op2', 'input'])

        op1_node = Node(graph, 'Op1')
        op1_node.add_output_port(0)
        self.assertTrue(op1_node.out_port(0).get_tensor_names() == [])

        op2_node = Node(graph, 'Op2')
        op2_node.add_output_port(0)
        self.assertTrue(op2_node.out_port(0).get_tensor_names() == [])

    def test_port_renumber(self):
        graph = build_graph(nodes, [('input', 'input_data'), ('input_data', 'Op1'),
                                    ('Op1', 'Op1_data', {'out': 1}), ('Op1_data', 'Op2')])
        input_node = Node(graph, 'input')
        self.assertTrue(input_node.out_port(0).get_tensor_names(port_renumber=True) == ['Op1\\,Op2', 'input'])

        op1_node = Node(graph, 'Op1')
        op1_node.add_output_port(0)

        self.assertTrue(op1_node.out_port(0).get_tensor_names(port_renumber=True) == ['Op1\\,Op2'])

    def test_reconnect_middle_case1(self):
        graph = build_graph(nodes, [('input', 'input_data'), ('input_data', 'Op1'), ('Op3', 'Op3_data')])
        input_node = Node(graph, 'input')

        input_node_out_port = input_node.out_port(0)
        self.assertTrue(input_node_out_port.get_tensor_names() == ['Op1\\,Op2', 'input'])

        op3_node = Node(graph, 'Op3')
        input_node_out_port.get_connection().set_source(op3_node.out_port(0))

        self.assertTrue(input_node_out_port.get_tensor_names() is None)
        self.assertTrue(op3_node.out_port(0).get_tensor_names() == ['Op1\\,Op2', 'Op3', 'input'])

    def test_reconnect_front_case1(self):
        graph = build_graph(nodes, [('input', 'Op1', {'in': 0, 'out': 0, 'fw_tensor_debug_info': [('input', 'input'),
                                                                                                  ('Op1', 'Op1,Op2')]}),
                                    ('Op3', 'Op2', {'in': 0, 'out': 0, 'fw_tensor_debug_info': [('Op3', 'Op3')]})])
        graph.stage = 'front'
        input_node = Node(graph, 'input')

        input_node_out_port = input_node.out_port(0)
        self.assertTrue(input_node_out_port.get_tensor_names() == ['Op1\\,Op2', 'input'])

        op3_node = Node(graph, 'Op3')
        input_node_out_port.get_connection().set_source(op3_node.out_port(0))

        self.assertTrue(input_node_out_port.get_tensor_names() == [])
        self.assertTrue(op3_node.out_port(0).get_tensor_names() == ['Op1\\,Op2', 'Op3', 'input'])

    def test_reconnect_middle_case1(self):
        graph = build_graph(nodes, [('input', 'input_data'), ('input_data', 'Op1'), ('Op3', 'Op3_data')])
        input_node = Node(graph, 'input')

        input_node_out_port = input_node.out_port(0)
        self.assertTrue(input_node_out_port.get_tensor_names() == ['Op1\\,Op2', 'input'])

        op3_node = Node(graph, 'Op3')
        input_node_out_port.get_connection().set_source(op3_node.out_port(0))

        self.assertTrue(input_node_out_port.get_tensor_names() == [])
        self.assertTrue(op3_node.out_port(0).get_tensor_names() == ['Op1\\,Op2', 'Op3', 'input'])

    def test_reconnect_middle_case2(self):
        graph = build_graph(nodes, [('input', 'input_data'), ('input_data', 'Op1', {'out': 0}),
                                    ('input_data', 'Op1', {'out': 1}), ('Op3', 'Op3_data')])
        input_node = Node(graph, 'input')

        input_node_out_port = input_node.out_port(0)
        self.assertTrue(input_node_out_port.get_tensor_names() == ['Op1\\,Op2', 'input'])

        op3_node = Node(graph, 'Op3')
        input_node_out_port.get_connection().set_source(op3_node.out_port(0))

        self.assertTrue(input_node_out_port.get_tensor_names() == [])
        self.assertTrue(op3_node.out_port(0).get_tensor_names() == ['Op1\\,Op2', 'Op3', 'input'])


class TestPortMethods(unittest.TestCase):

    def test_middle_disconnect_several_edges_between_two_nodes(self):
        graph = build_graph(nodes, [('input', 'input_data'), ('input_data', 'Op1'),
                                    ('Op1', 'Op1_data'), ('Op1_data', 'Op2', {'in': 0}), ('Op1_data', 'Op2', {'in': 1}),
                                    ('Op1_data', 'Op2', {'in': 2})],
                            nodes_with_edges_only=True)
        op1_node = Node(graph, 'Op1')
        op1_node.out_port(0).disconnect()
        self.assertTrue(op1_node.out_port(0).disconnected())


class TestForceShape(unittest.TestCase):
    def test_set_value_and_shape_with_force_shape_attribute_in_op(self):
        import numpy as np
        graph = build_graph({**valued_const_with_data('const', np.array([1, 2, 3])), **result()},
                            [*connect('const', 'output')])

        node = Node(graph, 'const')
        node['force_shape'] = np.array([2, 5, 7], dtype=np.int64)
        node.out_port(0).data.set_value(np.zeros(35))
        self.assertTrue(np.array_equal(node.out_port(0).data.get_shape(), np.array([2, 5, 7], dtype=np.int64)),
                        "node.out_port(0).data.get_shape()={} != [2, 5, 7]".format(node.out_port(0).data.get_shape()))

    def test_set_value_and_shape_with_force_shape_attribute_in_data(self):
        import numpy as np
        graph = build_graph({**valued_const_with_data('const', np.array([1, 2, 3])), **result()},
                            [*connect('const', 'output')])

        node = Node(graph, 'const')
        Node(graph, 'const_d')['force_shape'] = np.array([2, 5, 7], dtype=np.int64)
        node.out_port(0).data.set_value(np.zeros(30))
        self.assertTrue(np.array_equal(node.out_port(0).data.get_shape(), np.array([2, 5, 7], dtype=np.int64)),
                        "node.out_port(0).data.get_shape()={} != [2, 5, 7]".format(
                            node.out_port(0).data.get_shape()))

