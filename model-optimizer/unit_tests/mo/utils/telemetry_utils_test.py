# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from collections import Counter
from unittest.mock import Mock

from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Graph, Node
from mo.utils.telemetry_utils import send_op_names_info, send_shapes_info
from unit_tests.utils.graph import build_graph, regular_op

try:
    import openvino_telemetry as tm
except ImportError:
    import mo.utils.telemetry_stub as tm


class TestTelemetryUtils(unittest.TestCase):
    @staticmethod
    def init_telemetry_mocks():
        tm.Telemetry.__init__ = Mock(return_value=None)
        tm.Telemetry.send_event = Mock()

    def test_send_op_names_info(self):
        graph = Graph()
        graph.add_nodes_from(['node1'])
        graph.op_names_statistic = Counter(['a', 'a', 'a', 'b', 'b'])

        sub_graph1 = Graph()
        sub_graph1.add_nodes_from(['node2'])
        sub_graph1.op_names_statistic = Counter(['a', 'c', 'c'])

        sub_graph2 = Graph()
        sub_graph2.op_names_statistic = Counter(['a', 'd'])

        node1 = Node(graph, 'node1')
        node1['sub_graphs'] = ['sub_graph1']
        node1['sub_graph1'] = sub_graph1

        node2 = Node(sub_graph1, 'node2')
        node2['sub_graphs'] = ['sub_graph2']
        node2['sub_graph2'] = sub_graph2

        self.init_telemetry_mocks()

        send_op_names_info('framework', graph)
        tm.Telemetry.send_event.assert_any_call('mo', 'op_count', 'framework_a', 5)
        tm.Telemetry.send_event.assert_any_call('mo', 'op_count', 'framework_b', 2)
        tm.Telemetry.send_event.assert_any_call('mo', 'op_count', 'framework_c', 2)
        tm.Telemetry.send_event.assert_any_call('mo', 'op_count', 'framework_d', 1)

    def test_send_shapes_info(self):
        graph = build_graph({**regular_op('placeholder1', {'shape': int64_array([1, 3, 20, 20]), 'type': 'Parameter'}),
                             **regular_op('placeholder2', {'shape': int64_array([2, 4, 10]), 'type': 'Parameter'}),
                             **regular_op('mul', {'shape': int64_array([7, 8]), 'type': 'Multiply'})}, [])

        self.init_telemetry_mocks()

        send_shapes_info('framework', graph)
        tm.Telemetry.send_event.assert_any_call('mo', 'input_shapes', '{fw:framework,shape:"[ 1  3 20 20],[ 2  4 10]"}')
        tm.Telemetry.send_event.assert_any_call('mo', 'partially_defined_shape',
                                                '{partially_defined_shape:0,fw:framework}')

    def test_send_dynamic_shapes_case1(self):
        graph = build_graph({**regular_op('placeholder1', {'shape': int64_array([-1, 3, 20, 20]), 'type': 'Parameter'}),
                             **regular_op('mul', {'shape': int64_array([7, 8]), 'type': 'Multiply'})}, [])

        self.init_telemetry_mocks()

        send_shapes_info('framework', graph)
        tm.Telemetry.send_event.assert_any_call('mo', 'input_shapes', '{fw:framework,shape:"[-1  3 20 20]"}')
        tm.Telemetry.send_event.assert_any_call('mo', 'partially_defined_shape',
                                                '{partially_defined_shape:1,fw:framework}')

    def test_send_dynamic_shapes_case2(self):
        graph = build_graph({**regular_op('placeholder1', {'shape': int64_array([2, 3, 20, 20]), 'type': 'Parameter'}),
                             **regular_op('placeholder2', {'shape': int64_array([7, 4, 10]), 'type': 'Parameter'}),
                             **regular_op('placeholder3', {'shape': int64_array([5, 4, 0]), 'type': 'Parameter'}),
                             **regular_op('mul', {'shape': int64_array([7, 8]), 'type': 'Multiply'})}, [])

        self.init_telemetry_mocks()

        send_shapes_info('framework', graph)
        tm.Telemetry.send_event.assert_any_call('mo', 'input_shapes',
                                                '{fw:framework,shape:"[ 2  3 20 20],[ 7  4 10],[5 4 0]"}')
        tm.Telemetry.send_event.assert_any_call('mo', 'partially_defined_shape',
                                                '{partially_defined_shape:1,fw:framework}')
