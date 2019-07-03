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

from extensions.front.mxnet.ssd_pattern_remove_reshape import SsdPatternRemoveReshape
from mo.utils.unittest.graph import build_graph
from mo.graph.graph import Node


class TestSsdPatternRemoveReshape(unittest.TestCase):
    def test_pattern_remove_reshape(self):
        graph = build_graph({'node_1': {'type': 'Identity', 'kind': 'op', 'op': 'Placeholder'},
                             'node_2': {'type': 'Identity', 'kind': 'op'},
                             'node_multi_box_prior1': {'type': '_contrib_MultiBoxPrior', 'kind': 'op',
                                                       'op': '_contrib_MultiBoxPrior'},
                             'node_multi_box_prior2': {'type': '_contrib_MultiBoxPrior', 'kind': 'op',
                                                       'op': '_contrib_MultiBoxPrior'},
                             'node_multi_box_prior3': {'type': '_contrib_MultiBoxPrior', 'kind': 'op',
                                                       'op': '_contrib_MultiBoxPrior'},
                             'node_concat': {'type': 'Concat', 'kind': 'op', 'op': 'Concat'},
                             'node_reshape': {'type': 'Reshape', 'kind': 'op', 'op': 'Reshape'},
                             'node_3': {'type': 'Identity', 'kind': 'op'},
                             },
                            [('node_1', 'node_2'),
                             ('node_2', 'node_multi_box_prior1'),
                             ('node_2', 'node_multi_box_prior2'),
                             ('node_2', 'node_multi_box_prior3'),
                             ('node_multi_box_prior1', 'node_concat'),
                             ('node_multi_box_prior2', 'node_concat'),
                             ('node_multi_box_prior3', 'node_concat'),
                             ('node_concat', 'node_reshape'),
                             ('node_reshape', 'node_3'), ],
                            {
                                'node_concat': {'symbol_dict': {'attrs': {'dim': 3}}},
                            })

        pattern = SsdPatternRemoveReshape()
        pattern.find_and_replace_pattern(graph)
        node_concat = Node(graph, 'node_concat')
        self.assertEqual(node_concat['symbol_dict']['attrs']['dim'], 2)
        self.assertFalse(graph.has_node('node_reshape'))
        self.assertTrue(graph.has_edge(Node(graph, 'node_concat').id, Node(graph, 'node_3').id))
