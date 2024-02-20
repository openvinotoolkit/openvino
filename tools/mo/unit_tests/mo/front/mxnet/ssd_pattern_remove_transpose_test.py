# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.front.mxnet.ssd_pattern_remove_transpose import SsdPatternRemoveTranspose
from openvino.tools.mo.graph.graph import Node
from unit_tests.utils.graph import build_graph


class TestSsdPatternRemoveTranspose(unittest.TestCase):
    def test_pattern_remove_transpose(self):
        graph = build_graph({'node_1': {'type': 'Identity', 'value': None, 'kind': 'op', 'op': 'Parameter'},
                             'node_3': {'type': 'Identity', 'value': None, 'kind': 'op'},
                             'node_4': {'type': 'Identity', 'value': None, 'kind': 'op'},
                             'node_transpose': {'type': 'transpose', 'value': None, 'kind': 'op', 'op': 'Transpose'},
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
