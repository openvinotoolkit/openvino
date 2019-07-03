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

import numpy as np

from extensions.front.mxnet.ssd_reorder_detection_out_inputs import SsdReorderDetectionOutInputs
from mo.utils.unittest.graph import build_graph
from mo.graph.graph import Node


class TestSsdReorderDetectionOutInputs(unittest.TestCase):
    def test_reorder_detection_out_inputs(self):
        graph = build_graph(
            {'node_1': {'type': 'Identity', 'kind': 'op', 'op': 'Placeholder'},
             'node_2': {'type': 'Identity', 'kind': 'op', 'op': 'Placeholder'},
             'node_3': {'type': 'Identity', 'kind': 'op', 'op': 'Placeholder'},
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
