"""
 Copyright (c) 2017-2019 Intel Corporation

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

from extensions.front.mxnet.ssd_pattern_flatten_softmax_activation import SsdPatternFlattenSoftmaxActivation
from mo.utils.unittest.graph import build_graph
from mo.graph.graph import Node


class TestSsdPatternFlattenSoftmaxActivation(unittest.TestCase):
    def test_pattern_remove_transpose(self):
        graph = build_graph({'node_1': {'type': 'Identity', 'kind': 'op', 'op': 'Parameter'},
                             'node_2': {'type': 'Identity', 'kind': 'op'},
                             'node_3': {'type': 'Identity', 'kind': 'op'},
                             'node_softmax_activation': {'type': 'SoftMax', 'kind': 'op', 'op': 'SoftMax'},
                             'node_multi_box_detection': {'type': '_contrib_MultiBoxDetection', 'kind': 'op',
                                                          'op': '_contrib_MultiBoxDetection'},
                             'node_4': {'type': 'Identity', 'kind': 'op'},
                             },
                            [('node_1', 'node_softmax_activation'),
                             ('node_2', 'node_multi_box_detection'),
                             ('node_softmax_activation', 'node_multi_box_detection'),
                             ('node_3', 'node_multi_box_detection'),
                             ('node_multi_box_detection', 'node_4'), ],
                            )

        pattern = SsdPatternFlattenSoftmaxActivation()
        pattern.find_and_replace_pattern(graph)
        flatten_name = list(graph.nodes())[-1]
        self.assertTrue(graph.has_node(flatten_name))
        self.assertFalse(graph.has_edge(Node(graph, 'node_softmax_activation').id, Node(graph, 'node_multi_box_detection').id))
