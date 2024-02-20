# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.front.mxnet.ssd_pattern_flatten_softmax_activation import SsdPatternFlattenSoftmaxActivation
from openvino.tools.mo.graph.graph import Node
from unit_tests.utils.graph import build_graph


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
