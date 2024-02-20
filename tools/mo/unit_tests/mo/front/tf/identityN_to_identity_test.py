# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.front.tf.identityN_to_identity import IdentityN_to_Identity
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import result, regular_op_with_shaped_data, \
    regular_op_with_empty_data, build_graph, connect, empty_data

nodes = {
    **regular_op_with_shaped_data('placeholder_0', [1, 227, 227, 3], {'type': 'Parameter'}),
    **regular_op_with_shaped_data('placeholder_1', [1, 227, 227, 3], {'type': 'Parameter'}),

    **regular_op_with_empty_data('identityN', {'op': 'IdentityN', 'type': None, 'data_types': [np.int32, np.float32],
                                               'name': 'my_identity'}),
    **empty_data('identityN_1_d'),
    **regular_op_with_empty_data('identity0', {'op': 'Identity', 'type': None, 'data_type': np.int32,
                                               'name': 'my_identity/0_port'}),
    **regular_op_with_empty_data('identity1', {'op': 'Identity', 'type': None, 'data_type': np.float32,
                                               'name': 'my_identity/1_port'}),

    **result('output0'),
    **result('output1'),
}


class TestIdentityN(unittest.TestCase):
    def test_identityN(self):
        graph = build_graph(nodes, [
            *connect('placeholder_0', '0:identityN'),
            *connect('placeholder_1', '1:identityN'),
            *connect('identityN:0', 'output0'),
            ('identityN', 'identityN_1_d', {'out': 1}),
            ('identityN_1_d', 'output1', {'out': 1}),
        ], nodes_with_edges_only=True)

        IdentityN_to_Identity().find_and_replace_pattern(graph)

        graph_ref = build_graph(nodes, [
            *connect('placeholder_0', 'identity0'),
            *connect('placeholder_1', 'identity1'),
            *connect('identity0', 'output0'),
            *connect('identity1', 'output1'),
        ], nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output0', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_identityN_unused_ports(self):
            graph = build_graph(nodes, [
                *connect('placeholder_0', '0:identityN'),
                *connect('placeholder_1', '1:identityN'),
                *connect('identityN:0', 'output0'),
            ], nodes_with_edges_only=True)

            IdentityN_to_Identity().find_and_replace_pattern(graph)

            graph_ref = build_graph(nodes, [
                *connect('placeholder_0', 'identity0'),
                *connect('identity0', 'output0'),
            ], nodes_with_edges_only=True)

            (flag, resp) = compare_graphs(graph, graph_ref, 'output0', check_op_attrs=True)
            self.assertTrue(flag, resp)
