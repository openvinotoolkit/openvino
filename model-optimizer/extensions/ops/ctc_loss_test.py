"""
 Copyright (C) 2020 Intel Corporation

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

from extensions.ops.ctc_loss import CTCLoss
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph

nodes_attributes = {'logits': {'shape': None, 'value': None, 'kind': 'data'},
                    'logit_length': {'shape': None, 'value': None, 'kind': 'data'},
                    'labels': {'shape': None, 'value': None, 'kind': 'data'},
                    'label_length': {'shape': None, 'value': None, 'kind': 'data'},
                    'blank_index': {'shape': None, 'value': None, 'kind': 'data'},
                    'ctcloss_node': {'op': 'CTCLoss', 'kind': 'op', 'preprocess_collapse_repeated': False,
                                     'ctc_merge_repeated': True, 'unique': False},
                    'output': {'shape': None, 'value': None, 'kind': 'data'}}

# graph 1
edges1 = [('logits', 'ctcloss_node', {'in': 0}),
          ('logit_length', 'ctcloss_node', {'in': 1}),
          ('labels', 'ctcloss_node', {'in': 2}),
          ('label_length', 'ctcloss_node', {'in': 3}),
          ('blank_index', 'ctcloss_node', {'in': 4}),
          ('ctcloss_node', 'output', {'out': 0})]

# valid test case
inputs1 = {'logits': {'shape': int64_array([4, 100, 5])},
           'logit_length': {'shape': int64_array([4])},
           'labels': {'shape': int64_array([4, 100])},
           'label_length': {'shape': int64_array([4])},
           'blank_index': {'shape': int64_array([])}}

# invalid test case with incorrect rank for the second input tensor
inputs2 = {'logits': {'shape': int64_array([4, 100, 5])},
           'logit_length': {'shape': int64_array([4, 3])},
           'labels': {'shape': int64_array([4, 100])},
           'label_length': {'shape': int64_array([4])},
           'blank_index': {'shape': int64_array([])}}

# invalid test case with incorrect time dimension
inputs3 = {'logits': {'shape': int64_array([4, 100, 5])},
           'logit_length': {'shape': int64_array([4])},
           'labels': {'shape': int64_array([4, 300])},
           'label_length': {'shape': int64_array([4])},
           'blank_index': {'shape': int64_array([])}}

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
