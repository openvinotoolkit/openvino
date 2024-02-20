# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.ops.ctc_loss import CTCLoss
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.graph.graph import Node
from unit_tests.utils.graph import build_graph

nodes_attributes = {'logits': {'kind': 'op'},
                    'logits_data': {'shape': None, 'value': None, 'kind': 'data'},
                    'logit_length': {'kind': 'op'},
                    'logit_length_data': {'shape': None, 'value': None, 'kind': 'data'},
                    'labels': {'kind': 'op'},
                    'labels_data': {'shape': None, 'value': None, 'kind': 'data'},
                    'label_length': {'kind': 'op'},
                    'label_length_data': {'shape': None, 'value': None, 'kind': 'data'},
                    'blank_index': {'kind': 'op'},
                    'blank_index_data': {'shape': None, 'value': None, 'kind': 'data'},
                    'ctcloss_node': {'op': 'CTCLoss', 'kind': 'op', 'preprocess_collapse_repeated': False,
                                     'ctc_merge_repeated': True, 'unique': False},
                    'output': {'shape': None, 'value': None, 'kind': 'data'}}

# graph 1
edges1 = [('logits', 'logits_data'),
          ('logit_length', 'logit_length_data'),
          ('labels', 'labels_data'),
          ('label_length', 'label_length_data'),
          ('blank_index', 'blank_index_data'),
          ('logits_data', 'ctcloss_node', {'in': 0}),
          ('logit_length_data', 'ctcloss_node', {'in': 1}),
          ('labels_data', 'ctcloss_node', {'in': 2}),
          ('label_length_data', 'ctcloss_node', {'in': 3}),
          ('blank_index_data', 'ctcloss_node', {'in': 4}),
          ('ctcloss_node', 'output', {'out': 0})]

# valid test case
inputs1 = {'logits_data': {'shape': int64_array([4, 100, 5])},
           'logit_length_data': {'shape': int64_array([4])},
           'labels_data': {'shape': int64_array([4, 100])},
           'label_length_data': {'shape': int64_array([4])},
           'blank_index_data': {'shape': int64_array([])}}

# invalid test case with incorrect rank for the second input tensor
inputs2 = {'logits_data': {'shape': int64_array([4, 100, 5])},
           'logit_length_data': {'shape': int64_array([4, 3])},
           'labels_data': {'shape': int64_array([4, 100])},
           'label_length_data': {'shape': int64_array([4])},
           'blank_index_data': {'shape': int64_array([])}}

# invalid test case with incorrect time dimension
inputs3 = {'logits_data': {'shape': int64_array([4, 100, 5])},
           'logit_length_data': {'shape': int64_array([4])},
           'labels_data': {'shape': int64_array([4, 300])},
           'label_length_data': {'shape': int64_array([4])},
           'blank_index_data': {'shape': int64_array([])}}

class TestCTCLoss(unittest.TestCase):
    def test_infer1(self):
        graph = build_graph(nodes_attributes, edges1, inputs1)
        ctc_loss_node = Node(graph, 'ctcloss_node')
        CTCLoss.infer(ctc_loss_node)

        # prepare reference results
        ref_output_shape = int64_array([4])

        # get the result
        res_output_shape = graph.node['output']['shape']

        self.assertTrue(np.array_equal(ref_output_shape, res_output_shape),
                        'shapes do not match expected: {} and given: {}'.format(ref_output_shape, res_output_shape))

    def test_infer_invalid1(self):
        graph = build_graph(nodes_attributes, edges1, inputs2)
        ctc_loss_node = Node(graph, 'ctcloss_node')
        self.assertRaises(AssertionError, CTCLoss.infer, ctc_loss_node)

    def test_infer_invalid2(self):
        graph = build_graph(nodes_attributes, edges1, inputs3)
        ctc_loss_node = Node(graph, 'ctcloss_node')
        self.assertRaises(AssertionError, CTCLoss.infer, ctc_loss_node)
