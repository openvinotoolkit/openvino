"""
 Copyright (c) 2018 Intel Corporation

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

from extensions.front.caffe.axpy import AxpyToEltwise
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph_with_edge_attrs


class TestAxpyReplacer(unittest.TestCase):
    def test_axpy(self):
        nodes = {
            'node_1': {'kind': 'op', 'type': 'Identity', 'op': 'Placeholder'},
            'node_2': {'kind': 'op', 'type': 'Identity', 'op': 'Placeholder'},
            'node_3': {'kind': 'op', 'type': 'Identity', 'op': 'Placeholder'},
            'axpy': {'type': 'Axpy', 'kind': 'op', 'op': 'Axpy'},
            'node_4': {'kind': 'op', 'type': 'Identity', 'op': 'Placeholder'}}
        edges = [
            ('node_1', 'axpy', {'in': 0}),
            ('node_2', 'axpy', {'in': 1}),
            ('node_3', 'axpy', {'in': 2}),
            ('axpy', 'node_4', {'in': 0})]
        graph = build_graph_with_edge_attrs(nodes, edges)
        node = Node(graph, 'axpy')
        replacer = AxpyToEltwise()
        replacer.replace_op(graph, node)

        scale_node = [node for node, attrs in list(graph.nodes(data=True)) if attrs['type'] == 'ScaleShift']
        self.assertEqual(len(scale_node), 1)
        add_node = [node for node, attrs in list(graph.nodes(data=True)) if attrs['type'] == 'Eltwise']
        self.assertEqual(len(add_node), 1)
