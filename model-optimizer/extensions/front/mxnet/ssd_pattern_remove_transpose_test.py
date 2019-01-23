"""
 Copyright (c) 2017-2018 Intel Corporation

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

from extensions.front.mxnet.ssd_pattern_remove_transpose import SsdPatternRemoveTranspose
from mo.utils.unittest.graph import build_graph
from mo.graph.graph import Node


class TestSsdPatternRemoveTranspose(unittest.TestCase):
    def test_pattern_remove_transpose(self):
        graph = build_graph({'node_1': {'type': 'Identity', 'value': None, 'kind': 'op', 'op': 'Placeholder'},
                             'node_3': {'type': 'Identity', 'value': None, 'kind': 'op'},
                             'node_4': {'type': 'Identity', 'value': None, 'kind': 'op'},
                             'node_transpose': {'type': 'transpose', 'value': None, 'kind': 'op', 'op': 'transpose'},
                             'node_softmax_activation': {'type': 'SoftMax', 'value': None, 'kind': 'op',
                                                         'op': 'SoftMax'},
                             'node_multi_box_detection': {'type': '_contrib_MultiBoxDetection', 'value': None,
                                                          'kind': 'op', 'op': '_contrib_MultiBoxDetection'},
                             'node_5': {'type': 'Identity', 'value': None, 'kind': 'op'},
                             },
                            [('node_1', 'node_transpose'),
                             ('node_transpose', 'node_softmax_activation'),
                             ('node_3', 'node_multi_box_detection'),
                             ('node_softmax_activation', 'node_multi_box_detection'),
                             ('node_4', 'node_multi_box_detection'),
                             ('node_multi_box_detection', 'node_5'), ],
                            )

        pattern = SsdPatternRemoveTranspose()
        pattern.find_and_replace_pattern(graph)
        self.assertFalse(graph.has_node('node_transpose'))
        self.assertTrue(graph.has_edge(Node(graph, 'node_1').id, Node(graph, 'node_softmax_activation').id))
