"""
 Copyright (c) 2019 Intel Corporation

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

from extensions.front.tf.Unpack import Unpack
from mo.front.common.partial_infer.utils import int64_array
from mo.utils.unittest.graph import build_graph, compare_graphs


class UnpackReplacement(unittest.TestCase):
    def test_split_with_one_consumer(self):
        nodes = {
            'placeholder': {'op': 'Parameter', 'kind': 'op', 'shape': int64_array([2, 10, 4])},
            'unpack': {'op': 'Unpack', 'kind': 'op', 'axis': 2},
            'act_0': {'op': 'Sigmoid', 'kind': 'op'},
            'act_1': {'op': 'Sigmoid', 'kind': 'op'},
            'act_2': {'op': 'Sigmoid', 'kind': 'op'},
            'act_3': {'op': 'Sigmoid', 'kind': 'op'},
            'concat': {'op': 'Concat', 'kind': 'op', 'axis': 2},
            'op_output': {'op': 'OpOutput', 'kind': 'op'},
            'squeeze_0': {'op': 'Squeeze', 'kind': 'op'},
            'squeeze_1': {'op': 'Squeeze', 'kind': 'op'},
            'squeeze_2': {'op': 'Squeeze', 'kind': 'op'},
            'squeeze_3': {'op': 'Squeeze', 'kind': 'op'},
            'squeeze_0_dim': {'op': 'Const', 'kind': 'op', 'value': int64_array(2)},
            'squeeze_1_dim': {'op': 'Const', 'kind': 'op', 'value': int64_array(2)},
            'squeeze_2_dim': {'op': 'Const', 'kind': 'op', 'value': int64_array(2)},
            'squeeze_3_dim': {'op': 'Const', 'kind': 'op', 'value': int64_array(2)},
        }
        graph = build_graph(nodes, [
            ('placeholder', 'unpack', {'out': 0, 'in': 0}),
            ('unpack', 'act_0', {'out': 0, 'in': 0}),
            ('unpack', 'act_1', {'out': 1, 'in': 0}),
            ('unpack', 'act_2', {'out': 2, 'in': 0}),
            ('unpack', 'act_3', {'out': 3, 'in': 0}),
            ('act_0', 'concat', {'out': 0, 'in': 0}),
            ('act_1', 'concat', {'out': 0, 'in': 1}),
            ('act_2', 'concat', {'out': 0, 'in': 2}),
            ('act_3', 'concat', {'out': 0, 'in': 3}),
            ('concat', 'op_output', {'out': 1, 'in': 0}),
        ], nodes_with_edges_only=True)
        graph.stage = 'front'

        graph_ref = build_graph(nodes, [
            ('placeholder', 'unpack', {'out': 0, 'in': 0}),
            ('unpack', 'squeeze_0', {'out': 0, 'in': 0}),
            ('unpack', 'squeeze_1', {'out': 1, 'in': 0}),
            ('unpack', 'squeeze_2', {'out': 2, 'in': 0}),
            ('unpack', 'squeeze_3', {'out': 3, 'in': 0}),
            ('squeeze_0_dim', 'squeeze_0', {'out': 0, 'in': 1}),
            ('squeeze_1_dim', 'squeeze_1', {'out': 1, 'in': 1}),
            ('squeeze_2_dim', 'squeeze_2', {'out': 2, 'in': 1}),
            ('squeeze_3_dim', 'squeeze_3', {'out': 3, 'in': 1}),
            ('squeeze_0', 'act_0', {'out': 0, 'in': 0}),
            ('squeeze_1', 'act_1', {'out': 0, 'in': 0}),
            ('squeeze_2', 'act_2', {'out': 0, 'in': 0}),
            ('squeeze_3', 'act_3', {'out': 0, 'in': 0}),
            ('act_0', 'concat', {'out': 0, 'in': 0}),
            ('act_1', 'concat', {'out': 0, 'in': 1}),
            ('act_2', 'concat', {'out': 0, 'in': 2}),
            ('act_3', 'concat', {'out': 0, 'in': 3}),
            ('concat', 'op_output', {'out': 1, 'in': 0}),
        ], nodes_with_edges_only=True)

        Unpack().find_and_replace_pattern(graph=graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'op_output', check_op_attrs=True)
        self.assertTrue(flag, resp)
