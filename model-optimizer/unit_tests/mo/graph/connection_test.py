# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from mo.graph.graph import Node, Graph
from mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, regular_op

nodes = {
    **regular_op('input', {'type': 'Parameter'}),
    **regular_op('Op1', {'type': 'Op1', 'kind': 'op', 'op': 'Op1'}),
    **regular_op('Op2', {'type': 'Op2', 'kind': 'op', 'op': 'Op2'}),
    **regular_op('NewOp', {'type': 'NewOp', 'kind': 'op', 'op': 'NewOp'}),

    'input_data': {'kind': 'data', 'fw_tensor_debug_info': [('input', 'input')]},
    'Op1_data': {'kind': 'data', 'fw_tensor_debug_info': [('Op1', 'Op1')]},
    'Op2_data': {'kind': 'data', 'fw_tensor_debug_info': [('Op2', 'Op2')]},
    'NewOp_data': {'kind': 'data'},
}


class TestsFront(unittest.TestCase):
    def check_graph_attrs_front(self, graph: Graph, graph_ref: Graph):
        for node in graph_ref.get_op_nodes():
            if len(node.out_edges()) > 0:
                out_edge_ref = node.out_edge(0)
                out_edge = Node(graph, node.id).out_edge(0)
                if 'fw_tensor_debug_info' in out_edge_ref:
                    self.assertTrue(out_edge['fw_tensor_debug_info'] == out_edge_ref['fw_tensor_debug_info'])
                else:
                    self.assertFalse('fw_tensor_debug_info' in out_edge)

    def test_case1_merge(self):
        graph = build_graph(nodes,
                            [('input', 'Op1', {'in': 0, 'out': 0, 'fw_tensor_debug_info': [('input', 'input')]})])
        graph_ref = build_graph(nodes, [
            ('input', 'NewOp', {'in': 0, 'out': 0, 'fw_tensor_debug_info': [('input', 'input')]})])

        input_node = Node(graph, 'input')
        new_node = Node(graph, 'NewOp')
        new_node.add_input_port(0)

        graph.stage = 'front'
        new_node.in_port(0).get_connection().set_source(input_node.out_port(0), "merge")

        (flag, resp) = compare_graphs(graph, graph_ref, 'NewOp', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.check_graph_attrs_front(graph, graph_ref)

    def test_case1_source(self):
        graph = build_graph(nodes, [
            ('input', 'Op1', {'in': 0, 'out': 0, 'fw_tensor_debug_info': [('input', 'input')]})])
        graph_ref = build_graph(nodes, [
            ('input', 'NewOp', {'in': 0, 'out': 0, 'fw_tensor_debug_info': [('input', 'input')]})])

        input_node = Node(graph, 'input')
        new_node = Node(graph, 'NewOp')
        new_node.add_input_port(0)

        graph.stage = 'front'
        new_node.in_port(0).get_connection().set_source(input_node.out_port(0), "source")

        (flag, resp) = compare_graphs(graph, graph_ref, 'NewOp', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.check_graph_attrs_front(graph, graph_ref)

    def test_case1_dest(self):
        graph = build_graph(nodes, [
            ('input', 'Op1', {'in': 0, 'out': 0, 'fw_tensor_debug_info': [('input', 'input')]})])
        graph_ref = build_graph(nodes, [
            ('input', 'NewOp', {'in': 0, 'out': 0})])

        input_node = Node(graph, 'input')
        new_node = Node(graph, 'NewOp')
        new_node.add_input_port(0)

        graph.stage = 'front'
        new_node.in_port(0).get_connection().set_source(input_node.out_port(0), "dest")

        (flag, resp) = compare_graphs(graph, graph_ref, 'NewOp', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.check_graph_attrs_front(graph, graph_ref)

    def test_case2_merge(self):
        graph = build_graph(nodes,
                            [('input', 'Op1', {'in': 0, 'out': 0, 'fw_tensor_debug_info': [('input', 'input')]})])
        graph_ref = build_graph(nodes, [
            ('input', 'NewOp', {'in': 0, 'out': 0, 'fw_tensor_debug_info': [('input', 'input')]})])

        op1_node = Node(graph, 'Op1')
        new_node = Node(graph, 'NewOp')
        new_node.add_input_port(0)

        graph.stage = 'front'
        op1_node.in_port(0).get_connection().set_destination(new_node.in_port(0), "merge")

        (flag, resp) = compare_graphs(graph, graph_ref, 'NewOp', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.check_graph_attrs_front(graph, graph_ref)

    def test_case2_source(self):
        graph = build_graph(nodes,
                            [('input', 'Op1', {'in': 0, 'out': 0, 'fw_tensor_debug_info': [('input', 'input')]})])
        graph_ref = build_graph(nodes, [
            ('input', 'NewOp', {'in': 0, 'out': 0, 'fw_tensor_debug_info': [('input', 'input')]})])

        op1_node = Node(graph, 'Op1')
        new_node = Node(graph, 'NewOp')
        new_node.add_input_port(0)

        graph.stage = 'front'
        op1_node.in_port(0).get_connection().set_destination(new_node.in_port(0), "source")

        (flag, resp) = compare_graphs(graph, graph_ref, 'NewOp', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.check_graph_attrs_front(graph, graph_ref)

    def test_case2_dest(self):
        graph = build_graph(nodes,
                            [('input', 'Op1', {'in': 0, 'out': 0, 'fw_tensor_debug_info': [('input', 'input')]})])
        graph_ref = build_graph(nodes, [('input', 'NewOp', {'in': 0, 'out': 0})])

        op1_node = Node(graph, 'Op1')
        new_node = Node(graph, 'NewOp')
        new_node.add_input_port(0)

        graph.stage = 'front'
        op1_node.in_port(0).get_connection().set_destination(new_node.in_port(0), "dest")

        (flag, resp) = compare_graphs(graph, graph_ref, 'NewOp', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.check_graph_attrs_front(graph, graph_ref)

    def test_case3_merge(self):
        graph = build_graph(nodes,
                            [('input', 'Op1', {'in': 0, 'out': 0, 'fw_tensor_debug_info': [('input', 'input')]})])
        graph_ref = build_graph(nodes, [
            ('NewOp', 'Op1', {'in': 0, 'out': 0, 'fw_tensor_debug_info': [('input', 'input')]})])

        op1_node = Node(graph, 'Op1')
        new_node = Node(graph, 'NewOp')
        new_node.add_output_port(0)

        graph.stage = 'front'
        op1_node.in_port(0).get_connection().set_source(new_node.out_port(0), "merge")

        (flag, resp) = compare_graphs(graph, graph_ref, 'Op1', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.check_graph_attrs_front(graph, graph_ref)

    def test_case3_source(self):
        graph = build_graph(nodes,
                            [('input', 'Op1', {'in': 0, 'out': 0, 'fw_tensor_debug_info': [('input', 'input')]})])
        graph_ref = build_graph(nodes, [('NewOp', 'Op1', {'in': 0, 'out': 0})])

        op1_node = Node(graph, 'Op1')
        new_node = Node(graph, 'NewOp')
        new_node.add_output_port(0)

        graph.stage = 'front'
        op1_node.in_port(0).get_connection().set_source(new_node.out_port(0), "source")

        (flag, resp) = compare_graphs(graph, graph_ref, 'Op1', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.check_graph_attrs_front(graph, graph_ref)

    def test_case3_dest(self):
        graph = build_graph(nodes,
                            [('input', 'Op1', {'in': 0, 'out': 0, 'fw_tensor_debug_info': [('input', 'input')]})])
        graph_ref = build_graph(nodes, [
            ('NewOp', 'Op1', {'in': 0, 'out': 0, 'fw_tensor_debug_info': [('input', 'input')]})])

        op1_node = Node(graph, 'Op1')
        new_node = Node(graph, 'NewOp')
        new_node.add_output_port(0)

        graph.stage = 'front'
        op1_node.in_port(0).get_connection().set_source(new_node.out_port(0), "dest")

        (flag, resp) = compare_graphs(graph, graph_ref, 'Op1', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.check_graph_attrs_front(graph, graph_ref)

    def test_case4_merge(self):
        graph = build_graph(nodes,
                            [('input', 'Op1', {'in': 0, 'out': 0, 'fw_tensor_debug_info': [('input', 'input')]})])
        graph_ref = build_graph(nodes, [
            ('NewOp', 'Op1', {'in': 0, 'out': 0, 'fw_tensor_debug_info': [('input', 'input')]})])

        op1_node = Node(graph, 'Op1')
        new_node = Node(graph, 'NewOp')
        new_node.add_output_port(0)

        graph.stage = 'front'
        new_node.out_port(0).get_connection().set_destination(op1_node.in_port(0), "merge")

        (flag, resp) = compare_graphs(graph, graph_ref, 'Op1', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.check_graph_attrs_front(graph, graph_ref)

    def test_case4_source(self):
        graph = build_graph(nodes,
                            [('input', 'Op1', {'in': 0, 'out': 0, 'fw_tensor_debug_info': [('input', 0, 'input')]})])
        graph_ref = build_graph(nodes, [('NewOp', 'Op1', {'in': 0, 'out': 0})])

        op1_node = Node(graph, 'Op1')
        new_node = Node(graph, 'NewOp')
        new_node.add_output_port(0)

        graph.stage = 'front'
        new_node.out_port(0).get_connection().set_destination(op1_node.in_port(0), "source")

        (flag, resp) = compare_graphs(graph, graph_ref, 'Op1', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.check_graph_attrs_front(graph, graph_ref)

    def test_case4_dest(self):
        graph = build_graph(nodes,
                            [('input', 'Op1', {'in': 0, 'out': 0, 'fw_tensor_debug_info': [('input', 0, 'input')]})])
        graph_ref = build_graph(nodes, [
            ('NewOp', 'Op1', {'in': 0, 'out': 0, 'fw_tensor_debug_info': [('input', 0, 'input')]})])

        op1_node = Node(graph, 'Op1')
        new_node = Node(graph, 'NewOp')
        new_node.add_output_port(0)

        graph.stage = 'front'
        new_node.out_port(0).get_connection().set_destination(op1_node.in_port(0), "dest")

        (flag, resp) = compare_graphs(graph, graph_ref, 'Op1', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.check_graph_attrs_front(graph, graph_ref)

    def test_case5_merge(self):
        graph = build_graph(nodes,
                            [('input', 'Op1', {'in': 0, 'out': 0, 'fw_tensor_debug_info': [('input', 0, 'input')]}),
                             ('Op1', 'Op2', {'in': 0, 'out': 0, 'fw_tensor_debug_info': [('Op1', 0, 'Op1')]})])
        graph_ref = build_graph(nodes, [
            ('input', 'Op2', {'in': 0, 'out': 0, 'fw_tensor_debug_info': [('input', 0, 'input'), ('Op1', 0, 'Op1')]})])
        op1_node = Node(graph, 'Op1')

        inp_node = Node(graph, 'input')
        op2_node = Node(graph, 'Op2')
        graph.stage = 'front'
        op1_node.out_port(0).get_connection().set_source(op1_node.in_port(0).get_source(), "merge")

        (flag, resp) = compare_graphs(graph, graph_ref, 'Op2', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.check_graph_attrs_front(graph, graph_ref)

    def test_case5_source(self):
        graph = build_graph(nodes,
                            [('input', 'Op1', {'in': 0, 'out': 0, 'fw_tensor_debug_info': [('input', 0, 'input')]}),
                             ('Op1', 'Op2', {'in': 0, 'out': 0, 'fw_tensor_debug_info': [('Op1', 0, 'Op1')]})])
        graph_ref = build_graph(nodes, [
            ('input', 'Op2', {'in': 0, 'out': 0, 'fw_tensor_debug_info': [('input', 0, 'input')]})])
        op1_node = Node(graph, 'Op1')

        graph.stage = 'front'
        op1_node.out_port(0).get_connection().set_source(op1_node.in_port(0).get_source(), "source")

        (flag, resp) = compare_graphs(graph, graph_ref, 'Op2', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.check_graph_attrs_front(graph, graph_ref)

    def test_case5_dest(self):
        graph = build_graph(nodes,
                            [('input', 'Op1', {'in': 0, 'out': 0, 'fw_tensor_debug_info': [('input', 0, 'input')]}),
                             ('Op1', 'Op2', {'in': 0, 'out': 0, 'fw_tensor_debug_info': [('Op1', 0, 'Op1')]})])
        graph_ref = build_graph(nodes,
                                [('input', 'Op2', {'in': 0, 'out': 0, 'fw_tensor_debug_info': [('Op1', 0, 'Op1')]})])
        op1_node = Node(graph, 'Op1')

        graph.stage = 'front'
        op1_node.out_port(0).get_connection().set_source(op1_node.in_port(0).get_source(), "dest")

        (flag, resp) = compare_graphs(graph, graph_ref, 'Op2', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.check_graph_attrs_front(graph, graph_ref)

    def test_case6_merge(self):
        graph = build_graph(nodes,
                            [('input', 'Op1', {'in': 0, 'out': 0, 'fw_tensor_debug_info': [('input', 0, 'input')]}),
                             ('Op1', 'Op2', {'in': 0, 'out': 0, 'fw_tensor_debug_info': [('Op1', 0, 'Op1')]})])
        graph_ref = build_graph(nodes, [
            ('input', 'Op2', {'in': 0, 'out': 0, 'fw_tensor_debug_info': [('input', 0, 'input'), ('Op1', 0, 'Op1')]})])
        op1_node = Node(graph, 'Op1')

        graph.stage = 'front'
        op1_node.in_port(0).get_connection().set_destination(op1_node.out_port(0).get_destination(), "merge")

        (flag, resp) = compare_graphs(graph, graph_ref, 'Op2', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.check_graph_attrs_front(graph, graph_ref)

    def test_case6_source(self):
        graph = build_graph(nodes,
                            [('input', 'Op1', {'in': 0, 'out': 0, 'fw_tensor_debug_info': [('input', 0, 'input')]}),
                             ('Op1', 'Op2', {'in': 0, 'out': 0, 'fw_tensor_debug_info': [('Op1', 0, 'Op1')]})])
        graph_ref = build_graph(nodes, [
            ('input', 'Op2', {'in': 0, 'out': 0, 'fw_tensor_debug_info': [('input', 0, 'input')]})])
        op1_node = Node(graph, 'Op1')

        graph.stage = 'front'
        op1_node.in_port(0).get_connection().set_destination(op1_node.out_port(0).get_destination(), "source")

        (flag, resp) = compare_graphs(graph, graph_ref, 'Op2', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.check_graph_attrs_front(graph, graph_ref)

    def test_case6_dest(self):
        graph = build_graph(nodes,
                            [('input', 'Op1', {'in': 0, 'out': 0, 'fw_tensor_debug_info': [('input', 0, 'input')]}),
                             ('Op1', 'Op2', {'in': 0, 'out': 0, 'fw_tensor_debug_info': [('Op1', 0, 'Op1')]})])
        graph_ref = build_graph(nodes,
                                [('input', 'Op2', {'in': 0, 'out': 0, 'fw_tensor_debug_info': [('Op1', 0, 'Op1')]})])
        op1_node = Node(graph, 'Op1')

        graph.stage = 'front'
        op1_node.in_port(0).get_connection().set_destination(op1_node.out_port(0).get_destination(), "dest")

        (flag, resp) = compare_graphs(graph, graph_ref, 'Op2', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.check_graph_attrs_front(graph, graph_ref)


class TestsMiddle(unittest.TestCase):
    def check_graph_attrs_middle(self, graph: Graph, graph_ref: Graph):
        for node in graph_ref.get_op_nodes():
            if len(node.out_nodes()) > 0:
                data_node_ref = node.out_node(0)
                data_node = Node(graph, node.id).out_node(0)
                if 'fw_tensor_debug_info' in data_node_ref:
                    self.assertTrue(data_node_ref['fw_tensor_debug_info'] == data_node['fw_tensor_debug_info'])
                else:
                    self.assertFalse('fw_tensor_debug_info' in data_node)

    def test_case1_merge(self):
        graph = build_graph(nodes, [('input', 'input_data'), ('input_data', 'Op1')])
        graph_ref = build_graph(nodes, [('input', 'input_data'), ('input_data', 'Op1'),
                                        ('input_data', 'NewOp')])
        input_node = Node(graph, 'input')
        new_node = Node(graph, 'NewOp')
        new_node.add_input_port(0)
        new_node.in_port(0).get_connection().set_source(input_node.out_port(0), "merge")

        (flag, resp) = compare_graphs(graph, graph_ref, 'NewOp', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.check_graph_attrs_middle(graph, graph_ref)

    def test_case1_source(self):
        graph = build_graph(nodes, [('input', 'input_data'), ('input_data', 'Op1')])
        graph_ref = build_graph(nodes, [('input', 'input_data'), ('input_data', 'Op1'),
                                        ('input_data', 'NewOp')])
        input_node = Node(graph, 'input')
        new_node = Node(graph, 'NewOp')
        new_node.add_input_port(0)
        new_node.in_port(0).get_connection().set_source(input_node.out_port(0), "source")

        (flag, resp) = compare_graphs(graph, graph_ref, 'NewOp', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.check_graph_attrs_middle(graph, graph_ref)

    def test_case1_dest(self):
        graph = build_graph(nodes, [('input', 'input_data'), ('input_data', 'Op1')])
        graph_ref = build_graph(nodes, [('input', 'input_data'), ('input_data', 'Op1'),
                                        ('input_data', 'NewOp')])

        input_node_data = Node(graph_ref, 'input_data')
        del input_node_data['fw_tensor_debug_info']

        input_node = Node(graph, 'input')
        new_node = Node(graph, 'NewOp')
        new_node.add_input_port(0)
        new_node.in_port(0).get_connection().set_source(input_node.out_port(0), "dest")

        (flag, resp) = compare_graphs(graph, graph_ref, 'NewOp', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.check_graph_attrs_middle(graph, graph_ref)

    def test_case2_merge(self):
        graph = build_graph(nodes, [('input', 'input_data'), ('input_data', 'Op1')])
        graph_ref = build_graph(nodes, [('input', 'input_data'),
                                        ('input_data', 'NewOp')])
        op1_node = Node(graph, 'Op1')
        new_node = Node(graph, 'NewOp')
        new_node.add_input_port(0)
        op1_node.in_port(0).get_connection().set_destination(new_node.in_port(0), "merge")

        (flag, resp) = compare_graphs(graph, graph_ref, 'NewOp', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.check_graph_attrs_middle(graph, graph_ref)

    def test_case2_source(self):
        graph = build_graph(nodes, [('input', 'input_data'), ('input_data', 'Op1')])
        graph_ref = build_graph(nodes, [('input', 'input_data'),
                                        ('input_data', 'NewOp')])
        op1_node = Node(graph, 'Op1')
        new_node = Node(graph, 'NewOp')
        new_node.add_input_port(0)
        op1_node.in_port(0).get_connection().set_destination(new_node.in_port(0), "source")

        (flag, resp) = compare_graphs(graph, graph_ref, 'NewOp', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.check_graph_attrs_middle(graph, graph_ref)

    def test_case2_dest(self):
        graph = build_graph(nodes, [('input', 'input_data'), ('input_data', 'Op1')])
        graph_ref = build_graph(nodes, [('input', 'input_data'),
                                        ('input_data', 'NewOp')])

        input_node_data = Node(graph_ref, 'input_data')
        del input_node_data['fw_tensor_debug_info']

        op1_node = Node(graph, 'Op1')
        new_node = Node(graph, 'NewOp')
        new_node.add_input_port(0)
        op1_node.in_port(0).get_connection().set_destination(new_node.in_port(0), "dest")

        (flag, resp) = compare_graphs(graph, graph_ref, 'NewOp', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.check_graph_attrs_middle(graph, graph_ref)

    def test_case3_merge(self):
        graph = build_graph(nodes, [('input', 'input_data'), ('input_data', 'Op1')])
        graph_ref = build_graph(nodes, [('input', 'input_data'), ('NewOp', 'NewOp_data'), ('NewOp_data', 'Op1')])

        new_op_data = Node(graph_ref, 'NewOp_data')
        new_op_data['fw_tensor_debug_info'] = [('input', 'input')]

        input_data = Node(graph_ref, 'input_data')
        del input_data['fw_tensor_debug_info']

        op1_node = Node(graph, 'Op1')
        new_node = Node(graph, 'NewOp')
        new_node.add_output_port(0)
        op1_node.in_port(0).get_connection().set_source(new_node.out_port(0), "merge")

        (flag, resp) = compare_graphs(graph, graph_ref, 'Op1', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.check_graph_attrs_middle(graph, graph_ref)

    def test_case3_source(self):
        graph = build_graph(nodes, [('input', 'input_data'), ('input_data', 'Op1')])
        graph_ref = build_graph(nodes, [('input', 'input_data'), ('NewOp', 'NewOp_data'), ('NewOp_data', 'Op1')])

        op1_node = Node(graph, 'Op1')
        new_node = Node(graph, 'NewOp')
        new_node.add_output_port(0)
        op1_node.in_port(0).get_connection().set_source(new_node.out_port(0), "source")

        (flag, resp) = compare_graphs(graph, graph_ref, 'Op1', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.check_graph_attrs_middle(graph, graph_ref)

    def test_case3_dest(self):
        graph = build_graph(nodes, [('input', 'input_data'), ('input_data', 'Op1')])
        graph_ref = build_graph(nodes, [('input', 'input_data'), ('NewOp', 'NewOp_data'), ('NewOp_data', 'Op1')])

        new_op_data = Node(graph_ref, 'NewOp_data')
        new_op_data['fw_tensor_debug_info'] = [('input', 'input')]

        input_data = Node(graph_ref, 'input_data')
        del input_data['fw_tensor_debug_info']

        op1_node = Node(graph, 'Op1')
        new_node = Node(graph, 'NewOp')
        new_node.add_output_port(0)
        op1_node.in_port(0).get_connection().set_source(new_node.out_port(0), "dest")

        (flag, resp) = compare_graphs(graph, graph_ref, 'Op1', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.check_graph_attrs_middle(graph, graph_ref)

    def test_case4_merge(self):
        graph = build_graph(nodes, [('input', 'input_data'), ('input_data', 'Op1')])
        graph_ref = build_graph(nodes, [('input', 'input_data'), ('NewOp', 'NewOp_data'), ('NewOp_data', 'Op1')])

        new_op_data = Node(graph_ref, 'NewOp_data')
        new_op_data['fw_tensor_debug_info'] = [('input', 'input')]

        op1_node = Node(graph, 'Op1')
        new_node = Node(graph, 'NewOp')
        new_node.add_output_port(0)
        new_node.out_port(0).get_connection().set_destination(op1_node.in_port(0), "merge")

        (flag, resp) = compare_graphs(graph, graph_ref, 'Op1', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.check_graph_attrs_middle(graph, graph_ref)

    def test_case4_source(self):
        graph = build_graph(nodes, [('input', 'input_data'), ('input_data', 'Op1')])
        graph_ref = build_graph(nodes, [('input', 'input_data'), ('NewOp', 'NewOp_data'), ('NewOp_data', 'Op1')])

        op1_node = Node(graph, 'Op1')
        new_node = Node(graph, 'NewOp')
        new_node.add_output_port(0)
        new_node.out_port(0).get_connection().set_destination(op1_node.in_port(0), "source")

        (flag, resp) = compare_graphs(graph, graph_ref, 'Op1', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.check_graph_attrs_middle(graph, graph_ref)

    def test_case4_dest(self):
        graph = build_graph(nodes, [('input', 'input_data'), ('input_data', 'Op1')])
        graph_ref = build_graph(nodes, [('input', 'input_data'), ('NewOp', 'NewOp_data'), ('NewOp_data', 'Op1')])

        new_op_data = Node(graph_ref, 'NewOp_data')
        new_op_data['fw_tensor_debug_info'] = [('input', 'input')]

        op1_node = Node(graph, 'Op1')
        new_node = Node(graph, 'NewOp')
        new_node.add_output_port(0)
        new_node.out_port(0).get_connection().set_destination(op1_node.in_port(0), "dest")

        (flag, resp) = compare_graphs(graph, graph_ref, 'Op1', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.check_graph_attrs_middle(graph, graph_ref)

    def test_case5_merge(self):
        graph = build_graph(nodes, [('input', 'input_data'), ('input_data', 'Op1'),
                                    ('Op1', 'Op1_data'), ('Op1_data', 'Op2')])
        graph_ref = build_graph(nodes, [('input', 'input_data'), ('input_data', 'Op1'),
                                        ('Op1', 'Op1_data'), ('input_data', 'Op2')])

        input_data = Node(graph_ref, 'input_data')
        input_data['fw_tensor_debug_info'] = [('input', 'input'), ('Op1', 'Op1')]

        op1_data = Node(graph_ref, 'Op1_data')
        del op1_data['fw_tensor_debug_info']

        op1_node = Node(graph, 'Op1')
        op1_node.out_port(0).get_connection().set_source(op1_node.in_port(0).get_source(), "merge")

        (flag, resp) = compare_graphs(graph, graph_ref, 'Op2', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.check_graph_attrs_middle(graph, graph_ref)

    def test_case5_source(self):
        graph = build_graph(nodes, [('input', 'input_data'), ('input_data', 'Op1'),
                                    ('Op1', 'Op1_data'), ('Op1_data', 'Op2')])
        graph_ref = build_graph(nodes, [('input', 'input_data'), ('input_data', 'Op1'),
                                        ('Op1', 'Op1_data'), ('input_data', 'Op2')])

        input_data = Node(graph_ref, 'input_data')
        input_data['fw_tensor_debug_info'] = [('input', 'input')]

        op1_node = Node(graph, 'Op1')
        op1_node.out_port(0).get_connection().set_source(op1_node.in_port(0).get_source(), "source")

        (flag, resp) = compare_graphs(graph, graph_ref, 'Op2', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.check_graph_attrs_middle(graph, graph_ref)

    def test_case5_dest(self):
        graph = build_graph(nodes, [('input', 'input_data'), ('input_data', 'Op1'),
                                    ('Op1', 'Op1_data'), ('Op1_data', 'Op2')])
        graph_ref = build_graph(nodes, [('input', 'input_data'), ('input_data', 'Op1'),
                                        ('Op1', 'Op1_data'), ('input_data', 'Op2')])

        input_data = Node(graph_ref, 'input_data')
        input_data['fw_tensor_debug_info'] = [('Op1', 'Op1')]

        op1_data = Node(graph_ref, 'Op1_data')
        del op1_data['fw_tensor_debug_info']

        op1_node = Node(graph, 'Op1')
        op1_node.out_port(0).get_connection().set_source(op1_node.in_port(0).get_source(), "dest")

        (flag, resp) = compare_graphs(graph, graph_ref, 'Op2', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.check_graph_attrs_middle(graph, graph_ref)

    def test_case6_merge(self):
        graph = build_graph(nodes, [('input', 'input_data'), ('input_data', 'Op1'),
                                    ('Op1', 'Op1_data'), ('Op1_data', 'Op2')])
        graph_ref = build_graph(nodes, [('input', 'input_data'), ('input_data', 'Op2'),
                                        ('Op1', 'Op1_data')])

        input_data = Node(graph_ref, 'input_data')
        input_data['fw_tensor_debug_info'] = [('input', 'input'), ('Op1', 'Op1')]

        op1_node = Node(graph, 'Op1')
        op1_node.in_port(0).get_connection().set_destination(op1_node.out_port(0).get_destination(), "merge")

        (flag, resp) = compare_graphs(graph, graph_ref, 'Op2', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.check_graph_attrs_middle(graph, graph_ref)

    def test_case6_source(self):
        graph = build_graph(nodes, [('input', 'input_data'), ('input_data', 'Op1'),
                                    ('Op1', 'Op1_data'), ('Op1_data', 'Op2')])
        graph_ref = build_graph(nodes, [('input', 'input_data'), ('input_data', 'Op2'),
                                        ('Op1', 'Op1_data')])

        input_data = Node(graph_ref, 'input_data')
        input_data['fw_tensor_debug_info'] = [('input', 'input')]

        op1_node = Node(graph, 'Op1')
        op1_node.in_port(0).get_connection().set_destination(op1_node.out_port(0).get_destination(), "source")

        (flag, resp) = compare_graphs(graph, graph_ref, 'Op2', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.check_graph_attrs_middle(graph, graph_ref)

    def test_case6_dest(self):
        graph = build_graph(nodes, [('input', 'input_data'), ('input_data', 'Op1'),
                                    ('Op1', 'Op1_data'), ('Op1_data', 'Op2')])
        graph_ref = build_graph(nodes, [('input', 'input_data'), ('input_data', 'Op2'),
                                        ('Op1', 'Op1_data')])

        input_data = Node(graph_ref, 'input_data')
        input_data['fw_tensor_debug_info'] = [('Op1', 'Op1')]

        op1_node = Node(graph, 'Op1')
        op1_node.in_port(0).get_connection().set_destination(op1_node.out_port(0).get_destination(), "dest")

        (flag, resp) = compare_graphs(graph, graph_ref, 'Op2', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.check_graph_attrs_middle(graph, graph_ref)
