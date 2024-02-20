# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.front.mxnet.ssd_reorder_detection_out_inputs import SsdReorderDetectionOutInputs
from openvino.tools.mo.graph.graph import Node
from unit_tests.utils.graph import build_graph


class TestSsdReorderDetectionOutInputs(unittest.TestCase):
    def test_reorder_detection_out_inputs(self):
        graph = build_graph(
            {'node_1': {'type': 'Identity', 'kind': 'op', 'op': 'Parameter'},
             'node_2': {'type': 'Identity', 'kind': 'op', 'op': 'Parameter'},
             'node_3': {'type': 'Identity', 'kind': 'op', 'op': 'Parameter'},
             'multi_box_detection': {'type': '_contrib_MultiBoxDetection', 'kind': 'op',
                                     'op': '_contrib_MultiBoxDetection'},
             },
            [('node_1', 'multi_box_detection'),
             ('node_2', 'multi_box_detection'),
             ('node_3', 'multi_box_detection')],
            {
                'node_1': {'shape': np.array([1, 34928])},
                'node_2': {'shape': np.array([1, 183372])},
                'node_3': {'shape': np.array([1, 2, 34928])},
            })

        pattern = SsdReorderDetectionOutInputs()
        pattern.find_and_replace_pattern(graph)

        node_multi_box = Node(graph, 'multi_box_detection')

        node_input1 = node_multi_box.in_node(0)
        node_input2 = node_multi_box.in_node(1)
        node_input3 = node_multi_box.in_node(2)
        self.assertEqual(node_input1.name, 'node_2')
        self.assertEqual(node_input2.name, 'node_1')
        self.assertEqual(node_input3.name, 'node_3')
