# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.front.mxnet.ssd_pattern_remove_reshape import SsdPatternRemoveReshape
from openvino.tools.mo.graph.graph import Node
from unit_tests.utils.graph import build_graph


class TestSsdPatternRemoveReshape(unittest.TestCase):
    def test_pattern_remove_reshape(self):
        graph = build_graph({'node_1': {'type': 'Identity', 'kind': 'op', 'op': 'Parameter'},
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
        graph.stage = 'front'
        SsdPatternRemoveReshape().find_and_replace_pattern(graph)
        node_concat = Node(graph, 'node_concat')
        self.assertEqual(node_concat['symbol_dict']['attrs']['dim'], 2)
        self.assertFalse(graph.has_node('node_reshape'))
        self.assertTrue(graph.has_edge(Node(graph, 'node_concat').id, Node(graph, 'node_3').id))
