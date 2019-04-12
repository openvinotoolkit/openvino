"""
 Copyright (c) 2018-2019 Intel Corporation

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


class TestInstanceNormalization(unittest.TestCase):
    def test_default(self):
        nodes = {
            'input': {'kind': 'op', 'op': 'AnyOp'},
            'scale': {'kind': 'op', 'op': 'AnyOp'},
            'B': {'kind': 'op', 'op': 'AnyOp'},
            'node': {'kind': 'op', 'op': 'InstanceNormalization', 'epsilon': 0.123},
        }
        edges = [
            ('input', 'node'),
            ('scale', 'node'),
            ('B', 'node'),
        ]

        graph = build_graph(nodes, edges)
        tested_class = InstanceNormalization()
        tested_class.find_and_replace_pattern(graph)

        ref_nodes = {
            'input': {'op': 'AnyOp'},
            'scale': {'op': 'AnyOp'},
            'B': {'op': 'AnyOp'},
            'mvn': {'kind': 'op', 'op': 'MVN', 'name': 'node/InstanceNormalization/MVN', 'eps': 0.123},
            'mul': {'kind': 'op', 'op': 'Mul', 'name': 'node/InstanceNormalization/Mul'},
            'add': {'kind': 'op', 'op': 'Add', 'name': 'node/InstanceNormalization/Add'},
        }
        ref_edges = [
            ('input', 'mvn'),
            ('mvn', 'mul'),
            ('scale', 'mul'),
            ('mul', 'add'),
            ('B', 'add'),
        ]

        ref_graph = build_graph(ref_nodes, ref_edges)
        self.assertTrue(nx.is_isomorphic(graph, ref_graph, node_match))
