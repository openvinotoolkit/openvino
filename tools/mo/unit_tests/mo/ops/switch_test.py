# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import Mock, call

import numpy as np

from openvino.tools.mo.ops.switch import Switch
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph_with_edge_attrs, build_graph_with_attrs


class TestSwitch(unittest.TestCase):
    def test_switch_infer_with_condition(self):
        nodes = [
            ('tensor', {'value': np.zeros((3, 3)), 'kind': 'data', 'executable': True, 'shape': np.array([3, 3])}),
            ('pred_id', {'value': True, 'kind': 'data', 'executable': True}),
            ('switch', {'type': 'Switch', 'kind': 'op', 'op': 'Switch', 'infer': Switch.infer}),
            ('switch_data_0', {'value': None, 'kind': 'data', 'executable': True, 'shape': None}),
            ('switch_data_1', {'value': None, 'kind': 'data', 'executable': True, 'shape': None}),
            ('result_0', {'value': None, 'kind': 'op', 'executable': True, 'type': 'Result', 'op': 'Result'}),
            ('result_1', {'value': None, 'kind': 'op', 'executable': True, 'type': 'Result', 'op': 'Result'}),
        ]
        edges = [
            ('tensor', 'switch', {'in': 0}),
            ('pred_id', 'switch', {'in': 1}),
            ('switch', 'switch_data_0', {'out': 0}),
            ('switch', 'switch_data_1', {'out': 1}),
            ('switch_data_0', 'result_0'),
            ('switch_data_1', 'result_1'),
        ]
        graph = build_graph_with_attrs(nodes_with_attrs=nodes, edges_with_attrs=edges)

        # We should propagate shapes and values
        graph_ref = build_graph_with_attrs(nodes_with_attrs=nodes,
                                           edges_with_attrs=edges,
                                           update_nodes_attributes=[('switch_data_0', {'shape': np.array([3, 3]),
                                                                                       'value': np.zeros((3, 3))}),
                                                                    ('switch_data_1', {'shape': np.array([3, 3]),
                                                                                       'value': np.zeros((3, 3))})])

        node = Node(graph, 'switch')
        node.infer(node)

        (flag, resp) = compare_graphs(graph, graph_ref, 'switch_data_0', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_switch_infer_no_condition(self):
        nodes = [
            ('tensor', {'value': None, 'kind': 'data', 'executable': True, 'shape': np.array([1, 2, 1])}),
            ('pred_id', {'value': None, 'kind': 'data', 'executable': True}),
            ('switch', {'type': 'Switch', 'kind': 'op', 'op': 'Switch', 'infer': Switch.infer}),
            ('switch_data_0', {'value': None, 'kind': 'data', 'executable': True, 'shape': None}),
            ('switch_data_1', {'value': None, 'kind': 'data', 'executable': True, 'shape': None}),
            ('result_0', {'value': None, 'kind': 'op', 'executable': True, 'type': 'Result', 'op': 'Result'}),
            ('result_1', {'value': None, 'kind': 'op', 'executable': True, 'type': 'Result', 'op': 'Result'}),
        ]
        edges = [
            ('tensor', 'switch', {'in': 0}),
            ('pred_id', 'switch', {'in': 1}),
            ('switch', 'switch_data_0', {'out': 0}),
            ('switch', 'switch_data_1', {'out': 1}),
            ('switch_data_0', 'result_0'),
            ('switch_data_1', 'result_1'),
        ]
        graph = build_graph_with_attrs(nodes_with_attrs=nodes, edges_with_attrs=edges)

        # We should propagate only shapes
        graph_ref = build_graph_with_attrs(nodes_with_attrs=nodes,
                                           edges_with_attrs=edges,
                                           update_nodes_attributes=[('switch_data_0', {'shape': np.array([1, 2, 1])}),
                                                                    ('switch_data_1', {'shape': np.array([1, 2, 1])})])

        node = Node(graph, 'switch')
        node.infer(node)

        (flag, resp) = compare_graphs(graph, graph_ref, 'switch_data_0', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_switch_cf_infer_no_condition(self):
        me_mock = Mock()
        nodes = {
            'tensor': {'value': True, 'kind': 'data', 'executable': True},
            'pred_id': {'value': None, 'kind': 'data', 'executable': True},
            'switch': {'type': 'Switch', 'kind': 'op', 'op': 'Switch', 'control_flow_infer': Switch.control_flow_infer},
            'switch_data_0': {'value': None, 'kind': 'data', 'executable': True},
            'switch_data_1': {'value': None, 'kind': 'data', 'executable': True},
            'result_0': {'value': None, 'kind': 'op', 'executable': True, 'type': 'Result', 'op': 'Result'},
            'result_1': {'value': None, 'kind': 'op', 'executable': True, 'type': 'Result', 'op': 'Result'},
        }
        edges = [
            ('tensor', 'switch', {'in': 0}),
            ('pred_id', 'switch', {'in': 1}),
            ('switch', 'switch_data_0', {'out': 0}),
            ('switch', 'switch_data_1', {'out': 1}),
            ('switch_data_0', 'result_0', {'in': 0}),
            ('switch_data_1', 'result_1', {'in': 0}),
        ]
        graph = build_graph_with_edge_attrs(nodes, edges)

        node = Node(graph, 'switch')
        node.control_flow_infer(node, True, me_mock)
        # In this case we should mark all ports as executable
        me_mock.assert_has_calls([call('switch_data_0', True), call('switch_data_1', True)], any_order=True)

    def test_switch_cf_true_both_ports(self):
        me_mock = Mock()
        nodes = {
            'tensor': {'value': True, 'kind': 'data', 'executable': True},
            'pred_id': {'value': np.array(True), 'kind': 'data', 'executable': True},
            'switch': {'type': 'Switch', 'kind': 'op', 'op': 'Switch', 'control_flow_infer': Switch.control_flow_infer},
            'switch_data_0': {'value': None, 'kind': 'data', 'executable': True},
            'switch_data_1': {'value': None, 'kind': 'data', 'executable': True},
            'result_0': {'value': None, 'kind': 'op', 'executable': True, 'type': 'Result', 'op': 'Result'},
            'result_1': {'value': None, 'kind': 'op', 'executable': True, 'type': 'Result', 'op': 'Result'},
        }
        edges = [
            ('tensor', 'switch', {'in': 0}),
            ('pred_id', 'switch', {'in': 1}),
            ('switch', 'switch_data_0', {'out': 0}),
            ('switch', 'switch_data_1', {'out': 1}),
            ('switch_data_0', 'result_0', {'in': 0}),
            ('switch_data_1', 'result_1', {'in': 0}),
        ]
        graph = build_graph_with_edge_attrs(nodes, edges)
        node = Node(graph, 'switch')
        node.control_flow_infer(node, True, me_mock)
        me_mock.assert_has_calls([call('switch_data_0', False), call('switch_data_1', True)], any_order=True)

    def test_switch_cf_false_both_ports(self):
        me_mock = Mock()

        nodes = {
            'tensor': {'value': True, 'kind': 'data', 'executable': True},
            'pred_id': {'value': np.array(False), 'kind': 'data', 'executable': True},
            'switch': {'type': 'Switch', 'kind': 'op', 'op': 'Switch', 'control_flow_infer': Switch.control_flow_infer},
            'switch_data_0': {'value': None, 'kind': 'data', 'executable': True},
            'switch_data_1': {'value': None, 'kind': 'data', 'executable': True},
            'result_0': {'value': None, 'kind': 'op', 'executable': True, 'type': 'Result', 'op': 'Result'},
            'result_1': {'value': None, 'kind': 'op', 'executable': True, 'type': 'Result', 'op': 'Result'},
        }
        edges = [
            ('tensor', 'switch', {'in': 0}),
            ('pred_id', 'switch', {'in': 1}),
            ('switch', 'switch_data_0', {'out': 0}),
            ('switch', 'switch_data_1', {'out': 1}),
            ('switch_data_0', 'result_0', {'in': 0}),
            ('switch_data_1', 'result_1', {'in': 0}),
        ]
        graph = build_graph_with_edge_attrs(nodes, edges)
        node = Node(graph, 'switch')
        node.control_flow_infer(node, True, me_mock)
        me_mock.assert_has_calls([call('switch_data_0', True), call('switch_data_1', False)], any_order=True)

    def test_switch_cf_true_one_exec_port(self):
        me_mock = Mock()

        nodes = {
            'tensor': {'value': True, 'kind': 'data', 'executable': True},
            'pred_id': {'value': np.array(True), 'kind': 'data', 'executable': True},
            'switch': {'type': 'Switch', 'kind': 'op', 'op': 'Switch', 'control_flow_infer': Switch.control_flow_infer},
            'switch_data_1': {'value': None, 'kind': 'data', 'executable': True},
            'result_1': {'value': None, 'kind': 'op', 'executable': True, 'type': 'Result', 'op': 'Result'},
        }
        edges = [
            ('tensor', 'switch', {'in': 0}),
            ('pred_id', 'switch', {'in': 1}),
            ('switch', 'switch_data_1', {'out': 1}),
            ('switch_data_1', 'result_1', {'in': 0}),
        ]
        graph = build_graph_with_edge_attrs(nodes, edges)
        node = Node(graph, 'switch')
        node.control_flow_infer(node, True, me_mock)
        me_mock.assert_has_calls([call('switch_data_1', True)], any_order=True)

    def test_switch_cf_false_one_exec_port(self):
        me_mock = Mock()

        nodes = {
            'tensor': {'value': True, 'kind': 'data', 'executable': True},
            'pred_id': {'value': np.array(False), 'kind': 'data', 'executable': True},
            'switch': {'type': 'Switch', 'kind': 'op', 'op': 'Switch', 'control_flow_infer': Switch.control_flow_infer},
            'switch_data_0': {'value': None, 'kind': 'data', 'executable': True},
            'result_0': {'value': None, 'kind': 'op', 'executable': True, 'type': 'Result', 'op': 'Result'},
        }
        edges = [
            ('tensor', 'switch', {'in': 0}),
            ('pred_id', 'switch', {'in': 1}),
            ('switch', 'switch_data_0', {'out': 0}),
            ('switch_data_0', 'result_0', {'in': 0}),
        ]
        graph = build_graph_with_edge_attrs(nodes, edges)
        node = Node(graph, 'switch')
        node.control_flow_infer(node, True, me_mock)
        me_mock.assert_has_calls([call('switch_data_0', True)], any_order=True)

    def test_switch_cf_true_no_exec(self):
        me_mock = Mock()

        nodes = {
            'tensor': {'value': True, 'kind': 'data', 'executable': True},
            'pred_id': {'value':  np.array(True), 'kind': 'data', 'executable': True},
            'switch': {'type': 'Switch', 'kind': 'op', 'op': 'Switch', 'control_flow_infer': Switch.control_flow_infer},
            'switch_data_0': {'value': None, 'kind': 'data', 'executable': True},
            'result_0': {'value': None, 'kind': 'op', 'executable': True, 'type': 'Result', 'op': 'Result'},
        }
        edges = [
            ('tensor', 'switch', {'in': 0}),
            ('pred_id', 'switch', {'in': 1}),
            ('switch', 'switch_data_0', {'out': 0}),
            ('switch_data_0', 'result_0', {'in': 0}),
        ]
        graph = build_graph_with_edge_attrs(nodes, edges)
        node = Node(graph, 'switch')
        node.control_flow_infer(node, True, me_mock)
        me_mock.assert_has_calls([call('switch_data_0', False)], any_order=True)

    def test_switch_cf_false_no_exec(self):
        me_mock = Mock()

        nodes = {
            'tensor': {'value': True, 'kind': 'data', 'executable': True},
            'pred_id': {'value': np.array(False), 'kind': 'data', 'executable': True},
            'switch': {'type': 'Switch', 'kind': 'op', 'op': 'Switch', 'control_flow_infer': Switch.control_flow_infer},
            'switch_data_1': {'value': None, 'kind': 'data', 'executable': True},
            'result_1': {'value': None, 'kind': 'op', 'executable': True, 'type': 'Result', 'op': 'Result'},
        }
        edges = [
            ('tensor', 'switch', {'in': 0}),
            ('pred_id', 'switch', {'in': 1}),
            ('switch', 'switch_data_1', {'out': 1}),
            ('switch_data_1', 'result_1', {'in': 0}),
        ]
        graph = build_graph_with_edge_attrs(nodes, edges)
        node = Node(graph, 'switch')
        node.control_flow_infer(node, True, me_mock)
        me_mock.assert_has_calls([call('switch_data_1', False)], any_order=True)
