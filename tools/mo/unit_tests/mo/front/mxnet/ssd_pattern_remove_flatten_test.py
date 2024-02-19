# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.front.mxnet.ssd_pattern_remove_flatten import SsdPatternRemoveFlatten
from openvino.tools.mo.graph.graph import Node
from unit_tests.utils.graph import build_graph


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
