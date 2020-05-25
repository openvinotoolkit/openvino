"""
 Copyright (C) 2020 Intel Corporation

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

from extensions.front.tf.softplus import SoftPlus
from mo.front.common.partial_infer.utils import int64_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph


class TestSoftPlus(unittest.TestCase):
    nodes = {
        'node_1': {'shape': int64_array([1, 2, 3, 4]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
        'softplus': {'value': None, 'kind': 'op', 'op': 'Softplus'},
        'exp': {'value': None, 'kind': 'op', 'op': 'Exp'},
        'add': {'value': None, 'kind': 'op', 'op': 'Add'},
        'add_const': {'value': None, 'kind': 'op', 'op': 'Const'},
        'log': {'value': None, 'kind': 'op', 'op': 'Log'},
        'last': {'type': None, 'value': None, 'kind': 'op', 'op': 'Result'}
    }

    def test_softplus_1(self):
        graph = build_graph(self.nodes, [('node_1', 'softplus'),
                                         ('softplus', 'last')], nodes_with_edges_only=True)

        graph_ref = build_graph(self.nodes, [('node_1', 'exp'),
                                             ('exp', 'add'),
                                             ('add_const', 'add'),
                                             ('add', 'log'),
                                             ('log', 'last')], nodes_with_edges_only=True)

        graph.stage = 'front'
        SoftPlus().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'last', check_op_attrs=True)
        self.assertTrue(flag, resp)
