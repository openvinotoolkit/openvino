# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import Mock, call

import numpy as np

from extensions.ops.switch import Switch
from mo.graph.graph import Node
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph_with_edge_attrs, build_graph_with_attrs


class TestSwitch(unittest.TestCase):
    def test_switch_infer_with_condition(self):
        nodes = [
            ('tensor', {'value': np.zeros((3, 3)), 'kind': 'data', 'executable': True, 'shape': np.array([3, 3])}),
            ('pred_id', {'value': True, 'kind': 'data', 'executable': True}),
            ('switch', {'type': 'Switch', 'kind': 'op', 'op': 'Switch'}),
            ('switch_data_0', {'value': None, 'kind': 'data', 'executable': True}),
            ('switch_data_1', {'value': None, 'kind': 'data', 'executable': True})
        ]
        edges = [
            ('tensor', 'switch', {'in': 0}),
            ('pred_id', 'switch', {'in': 1}),
            ('switch', 'switch_data_0', {'out': 0}),
            ('switch', 'switch_data_1', {'out': 1})
        ]
        graph = build_graph_with_attrs(nodes_with_attrs=nodes, edges_with_attrs=edges)

        # We should propagate shapes and values
        graph_ref = build_graph_with_attrs(nodes_with_attrs=nodes,
                                           edges_with_attrs=edges,
                                           update_nodes_attributes=[('switch_data_0', {'shape': np.array([3, 3]),
                                                                                       'value': np.zeros((3,3))}),
                                                                    ('switch_data_1', {'shape': np.array([3, 3]),
                                                                                       'value': np.zeros((3,3))})])

        tested_class = Switch(graph=graph, attrs={})

        node = Node(graph, 'switch')
        tested_class.infer(node)

        (flag, resp) = compare_graphs(graph, graph_ref, 'switch_data_0', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_switch_infer_no_condition(self):
        nodes = [
            ('tensor', {'value': None, 'kind': 'data', 'executable': True, 'shape': np.array([1, 2, 1])}),
            ('pred_id', {'value': None, 'kind': 'data', 'executable': True}),
            ('switch', {'type': 'Switch', 'kind': 'op', 'op': 'Switch'}),
            ('switch_data_0', {'value': None, 'kind': 'data', 'executable': True}),
            ('switch_data_1', {'value': None, 'kind': 'data', 'executable': True})
        ]
        edges = [
            ('tensor', 'switch', {'in': 0}),
            ('pred_id', 'switch', {'in': 1}),
            ('switch', 'switch_data_0', {'out': 0}),
            ('switch', 'switch_data_1', {'out': 1})
        ]
        graph = build_graph_with_attrs(nodes_with_attrs=nodes, edges_with_attrs=edges)

        # We should propagate only shapes
        graph_ref = build_graph_with_attrs(nodes_with_attrs=nodes,
                                           edges_with_attrs=edges,
                                           update_nodes_attributes=[('switch_data_0', {'shape': np.array([1, 2, 1])}),
                                                                    ('switch_data_1', {'shape': np.array([1, 2, 1])})])

        tested_class = Switch(graph=graph, attrs={})

        node = Node(graph, 'switch')
        tested_class.infer(node)

        (flag, resp) = compare_graphs(graph, graph_ref, 'switch_data_0', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_switch_cf_infer_no_condition(self):
        me_mock = Mock()
        nodes = {
            'tensor': {'value': True, 'kind': 'data', 'executable': True},
            'pred_id': {'value': None, 'kind': 'data', 'executable': True},
            'switch': {'type': 'Switch', 'kind': 'op', 'op': 'Switch'},
            'switch_data_0': {'value': None, 'kind': 'data', 'executable': True},
            'switch_data_1': {'value': None, 'kind': 'data', 'executable': True}
        }
        edges = [
            ('tensor', 'switch', {'in': 0}),
            ('pred_id', 'switch', {'in': 1}),
            ('switch', 'switch_data_0', {'out': 0}),
            ('switch', 'switch_data_1', {'out': 1})
        ]
        graph = build_graph_with_edge_attrs(nodes, edges)

        tested_class = Switch(graph=graph, attrs={})
        node = Node(graph, 'switch')
        tested_class.control_flow_infer(node, True, me_mock)
        # In this case we should mark all ports as executable
        me_mock.assert_has_calls([call('switch_data_0', True), call('switch_data_1', True)], any_order=True)

    def test_switch_cf_true_both_ports(self):
        me_mock = Mock()
        nodes = {
            'tensor': {'value': True, 'kind': 'data', 'executable': True},
            'pred_id': {'value': np.array(True), 'kind': 'data', 'executable': True},
            'switch': {'type': 'Switch', 'kind': 'op', 'op': 'Switch'},
            'switch_data_0': {'value': None, 'kind': 'data', 'executable': True},
            'switch_data_1': {'value': None, 'kind': 'data', 'executable': True}
        }
        edges = [
            ('tensor', 'switch', {'in': 0}),
            ('pred_id', 'switch', {'in': 1}),
            ('switch', 'switch_data_0', {'out': 0}),
            ('switch', 'switch_data_1', {'out': 1})
        ]
        graph = build_graph_with_edge_attrs(nodes, edges)
        tested_class = Switch(graph=graph, attrs={})
        node = Node(graph, 'switch')
        tested_class.control_flow_infer(node, True, me_mock)
        me_mock.assert_has_calls([call('switch_data_0', False), call('switch_data_1', True)], any_order=True)

    def test_switch_cf_false_both_ports(self):
        me_mock = Mock()

        nodes = {
            'tensor': {'value': True, 'kind': 'data', 'executable': True},
            'pred_id': {'value': np.array(False), 'kind': 'data', 'executable': True},
            'switch': {'type': 'Switch', 'kind': 'op', 'op': 'Switch'},
            'switch_data_0': {'value': None, 'kind': 'data', 'executable': True},
            'switch_data_1': {'value': None, 'kind': 'data', 'executable': True}
        }
        edges = [
            ('tensor', 'switch', {'in': 0}),
            ('pred_id', 'switch', {'in': 1}),
            ('switch', 'switch_data_0', {'out': 0}),
            ('switch', 'switch_data_1', {'out': 1})
        ]
        graph = build_graph_with_edge_attrs(nodes, edges)
        tested_class = Switch(graph=graph, attrs={})
        node = Node(graph, 'switch')
        tested_class.control_flow_infer(node, True, me_mock)
        me_mock.assert_has_calls([call('switch_data_0', True), call('switch_data_1', False)], any_order=True)

    def test_switch_cf_true_one_exec_port(self):
        me_mock = Mock()

        nodes = {
            'tensor': {'value': True, 'kind': 'data', 'executable': True},
            'pred_id': {'value': np.array(True), 'kind': 'data', 'executable': True},
            'switch': {'type': 'Switch', 'kind': 'op', 'op': 'Switch'},
            'switch_data_1': {'value': None, 'kind': 'data', 'executable': True}
        }
        edges = [
            ('tensor', 'switch', {'in': 0}),
            ('pred_id', 'switch', {'in': 1}),
            ('switch', 'switch_data_1', {'out': 1})
        ]
        graph = build_graph_with_edge_attrs(nodes, edges)
        tested_class = Switch(graph=graph, attrs={})
        node = Node(graph, 'switch')
        tested_class.control_flow_infer(node, True, me_mock)
        me_mock.assert_has_calls([call('switch_data_1', True)], any_order=True)

    def test_switch_cf_false_one_exec_port(self):
        me_mock = Mock()

        nodes = {
            'tensor': {'value': True, 'kind': 'data', 'executable': True},
            'pred_id': {'value': np.array(False), 'kind': 'data', 'executable': True},
            'switch': {'type': 'Switch', 'kind': 'op', 'op': 'Switch'},
            'switch_data_0': {'value': None, 'kind': 'data', 'executable': True},
        }
        edges = [
            ('tensor', 'switch', {'in': 0}),
            ('pred_id', 'switch', {'in': 1}),
            ('switch', 'switch_data_0', {'out': 0}),
        ]
        graph = build_graph_with_edge_attrs(nodes, edges)
        tested_class = Switch(graph=graph, attrs={})
        node = Node(graph, 'switch')
        tested_class.control_flow_infer(node, True, me_mock)
        me_mock.assert_has_calls([call('switch_data_0', True)], any_order=True)

    def test_switch_cf_true_no_exec(self):
        me_mock = Mock()

        nodes = {
            'tensor': {'value': True, 'kind': 'data', 'executable': True},
            'pred_id': {'value':  np.array(True), 'kind': 'data', 'executable': True},
            'switch': {'type': 'Switch', 'kind': 'op', 'op': 'Switch'},
            'switch_data_0': {'value': None, 'kind': 'data', 'executable': True}
        }
        edges = [
            ('tensor', 'switch', {'in': 0}),
            ('pred_id', 'switch', {'in': 1}),
            ('switch', 'switch_data_0', {'out': 0}),
        ]
        graph = build_graph_with_edge_attrs(nodes, edges)
        tested_class = Switch(graph=graph, attrs={})
        node = Node(graph, 'switch')
        tested_class.control_flow_infer(node, True, me_mock)
        me_mock.assert_has_calls([call('switch_data_0', False)], any_order=True)

    def test_switch_cf_false_no_exec(self):
        me_mock = Mock()

        nodes = {
            'tensor': {'value': True, 'kind': 'data', 'executable': True},
            'pred_id': {'value': np.array(False), 'kind': 'data', 'executable': True},
            'switch': {'type': 'Switch', 'kind': 'op', 'op': 'Switch'},
            'switch_data_1': {'value': None, 'kind': 'data', 'executable': True}
        }
        edges = [
            ('tensor', 'switch', {'in': 0}),
            ('pred_id', 'switch', {'in': 1}),
            ('switch', 'switch_data_1', {'out': 1})
        ]
        graph = build_graph_with_edge_attrs(nodes, edges)
        tested_class = Switch(graph=graph, attrs={})
        node = Node(graph, 'switch')
        tested_class.control_flow_infer(node, True, me_mock)
        me_mock.assert_has_calls([call('switch_data_1', False)], any_order=True)
