"""
 Copyright (c) 2018-2019 Intel Corporation

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
from argparse import Namespace

from mo.graph.graph import Node
from extensions.front.mxnet.add_input_data_to_prior_boxes import AddInputDataToPriorBoxes
from mo.utils.unittest.graph import build_graph


class TestMxnetPipeline(unittest.TestCase):
    def test_mxnet_pipeline_1(self):
        graph = build_graph(
            {'data': {'type': 'Identity', 'value': None, 'kind': 'op', 'op': 'Placeholder'},
             'node_2': {'type': 'Identity', 'value': None, 'kind': 'op'},
             'node_multi_box': {'type': '_contrib_MultiBoxPrior', 'kind': 'op', 'op': '_contrib_MultiBoxPrior'},
             },
            [('data', 'node_2'),
             ('node_2', 'node_multi_box')],
            {
                'data': {'shape': np.array([1, 3, 227, 227])},
                'node_2': {'shape': np.array([1, 3, 10, 10])},
            })

        graph.graph['cmd_params'] = Namespace(input=None)
        AddInputDataToPriorBoxes().find_and_replace_pattern(graph)
        node_multi_box = Node(graph, 'node_multi_box')

        node_input1 = node_multi_box.in_node(0)
        node_input2 = node_multi_box.in_node(1)
        self.assertEqual(node_input1.name, 'node_2')
        self.assertEqual(node_input2.name, 'data')

    def test_mxnet_pipeline_2(self):
        graph = build_graph(
            {'node_1': {'type': 'Identity', 'value': None, 'kind': 'op', 'op': 'Placeholder'},
             'node_2': {'type': 'Identity', 'value': None, 'kind': 'op'},
             'node_multi_box': {'type': '_contrib_MultiBoxPrior', 'kind': 'op', 'op': '_contrib_MultiBoxPrior'},
             },
            [('node_1', 'node_2'),
             ('node_2', 'node_multi_box')],
            {
                'node_1': {'shape': np.array([1, 3, 227, 227])},
                'node_2': {'shape': np.array([1, 3, 10, 10])},
            })

        graph.graph['cmd_params'] = Namespace(input='node_1')
        AddInputDataToPriorBoxes().find_and_replace_pattern(graph)
        node_multi_box = Node(graph, 'node_multi_box')

        node_input1 = node_multi_box.in_node(0)
        node_input2 = node_multi_box.in_node(1)
        self.assertEqual(node_input1.name, 'node_2')
        self.assertEqual(node_input2.name, 'node_1')
