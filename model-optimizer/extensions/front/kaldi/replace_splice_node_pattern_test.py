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

from extensions.front.kaldi.replace_splice_node_pattern import ReplaceSpliceNodePattern
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph


class ReplaceSpliceNodePatternTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nodes_attributes = {
            'in_node': {'kind': 'op', 'op': 'Input', 'shape': [1, 13]},
            'slice': {'kind': 'op', 'op': 'Splice', 'context': range(-5, 5)}
        }
        cls.graph = build_graph(cls.nodes_attributes,
                                [('in_node', 'slice')])

        ReplaceSpliceNodePattern().find_and_replace_pattern(cls.graph)

    def test_memory(self):
        memory_nodes = [node for node in self.graph.nodes(data=True) if node[1]['op'] == 'Memory']
        self.assertEqual(len(memory_nodes), 2)
        for memory_node in memory_nodes:
            node = Node(self.graph, memory_node[0])
            if len(node.in_nodes()):
                self.assertEqual(node.index, 0)
            elif len(node.out_nodes()):
                self.assertEqual(node.index, 1)
        self.assertEqual(memory_nodes[0][1]['id'], memory_nodes[1][1]['id'])

    def test_crop(self):
        crop_node = [node for node in self.graph.nodes(data=True) if node[1]['op'] == 'Crop']
        self.assertEqual(len(crop_node), 1)
        crop_node = Node(self.graph, crop_node[0][0])
        self.assertEqual(crop_node.offset, [13])
        self.assertEqual(crop_node.dim, [13 * 9])

    def test_concat(self):
        concat_node = [node for node in self.graph.nodes(data=True) if node[1]['op'] == 'Concat']
        self.assertEqual(len(concat_node), 1)
        crop_node = Node(self.graph, concat_node[0][0])
        self.assertEqual(crop_node.axis, 1)
