# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from argparse import Namespace

import numpy as np

from openvino.tools.mo.front.mxnet.add_input_data_to_prior_boxes import AddInputDataToPriorBoxes
from openvino.tools.mo.graph.graph import Node
from unit_tests.utils.graph import build_graph


class TestMxnetPipeline(unittest.TestCase):
    def test_mxnet_pipeline_1(self):
        graph = build_graph(
            {'data': {'type': 'Identity', 'value': None, 'kind': 'op', 'op': 'Parameter'},
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
            {'node_1': {'type': 'Identity', 'value': None, 'kind': 'op', 'op': 'Parameter'},
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
