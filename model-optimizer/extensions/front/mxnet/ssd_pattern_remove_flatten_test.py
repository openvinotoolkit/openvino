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

from extensions.front.mxnet.ssd_pattern_remove_flatten import SsdPatternRemoveFlatten
from mo.utils.unittest.graph import build_graph
from mo.graph.graph import Node


class TestSsdPatternRemoveFlatten(unittest.TestCase):
    def test_pattern_remove_transpose(self):
        graph = build_graph({'node_1': {'type': 'Identity', 'kind': 'op', 'op': 'Parameter'},
                             'node_2': {'type': 'Identity', 'kind': 'op'},
                             'node_multi_box_prior': {'type': '_contrib_MultiBoxPrior', 'kind': 'op',
                                                      'op': '_contrib_MultiBoxPrior'},
                             'node_flatten': {'type': 'Flatten', 'kind': 'op', 'op': 'Flatten'},
                             'node_3': {'type': 'Identity', 'kind': 'op'},
                             },
                            [('node_1', 'node_2'),
                             ('node_2', 'node_multi_box_prior'),
                             ('node_multi_box_prior', 'node_flatten'),
                             ('node_flatten', 'node_3'), ],
                            )

        pattern = SsdPatternRemoveFlatten()
        pattern.find_and_replace_pattern(graph)
        self.assertFalse(graph.has_node('node_flatten'))
        self.assertTrue(graph.has_edge(Node(graph, 'node_multi_box_prior').id, Node(graph, 'node_3').id))
