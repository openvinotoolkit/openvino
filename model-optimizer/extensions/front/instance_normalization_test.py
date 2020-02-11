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

import networkx as nx

from extensions.front.instance_normalization import InstanceNormalization
from mo.utils.unittest.graph import build_graph
from mo.middle.pattern_match import node_match

nodes_attributes = {
    'input': {'kind': 'op', 'op': 'AnyOp'},
    'scale': {'kind': 'op', 'op': 'AnyOp'},
    'B': {'kind': 'op', 'op': 'AnyOp'},
    'node': {'kind': 'op', 'op': 'InstanceNormalization', 'epsilon': None},
}

nodes_ref_attributes = {
    'input': {'op': 'AnyOp'},
    'scale': {'op': 'AnyOp'},
    'B': {'op': 'AnyOp'},
    'mvn': {'kind': 'op', 'op': 'MVN', 'name': 'node/Ins_Norm/MVN_', 'eps': None},
    'mul': {'kind': 'op', 'op': 'Mul', 'name': 'node/Ins_Norm/mul_'},
    'add': {'kind': 'op', 'op': 'Add', 'name': 'node/Ins_Norm/add_'},
}


class TestInstanceNormalization(unittest.TestCase):
    def test_instance_normalization_test_1(self):
        graph = build_graph(nodes_attributes,
                            [('input', 'node'),
                             ('scale', 'node'),
                             ('B', 'node'),
                             ],
                            {'node': {'epsilon': 0.123},
                             }, nodes_with_edges_only=True)

        ref_graph = build_graph(nodes_ref_attributes,
                                [('input', 'mvn'),
                                 ('mvn', 'mul'),
                                 ('scale', 'mul'),
                                 ('mul', 'add'),
                                 ('B', 'add'),
                                 ],
                                {'mvn': {'eps': 0.123},
                                 }, nodes_with_edges_only=True)

        graph.stage = 'front'

        tested_class = InstanceNormalization()
        tested_class.find_and_replace_pattern(graph)

        self.assertTrue(nx.is_isomorphic(graph, ref_graph, node_match))
