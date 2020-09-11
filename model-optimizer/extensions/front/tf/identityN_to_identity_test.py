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

import numpy as np

from extensions.front.tf.identityN_to_identity import IdentityN_to_Identity
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import result, regular_op_with_shaped_data, \
    regular_op_with_empty_data, build_graph, connect, empty_data

nodes = {
    **regular_op_with_shaped_data('placeholder_0', [1, 227, 227, 3], {'type': 'Parameter'}),
    **regular_op_with_shaped_data('placeholder_1', [1, 227, 227, 3], {'type': 'Parameter'}),

    **regular_op_with_empty_data('identityN', {'op': 'IdentityN', 'type': None, 'data_types': [np.int32, np.float],
                                               'name': 'my_identity'}),
    **empty_data('identityN_1_d'),
    **regular_op_with_empty_data('identity0', {'op': 'Identity', 'type': None, 'data_type': np.int32,
                                               'name': 'my_identity/0_port'}),
    **regular_op_with_empty_data('identity1', {'op': 'Identity', 'type': None, 'data_type': np.float,
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
