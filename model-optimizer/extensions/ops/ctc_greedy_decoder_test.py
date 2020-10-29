"""
 Copyright (C) 2018-2020 Intel Corporation

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

from extensions.ops.ctc_greedy_decoder import CTCGreedyDecoderOp
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph


nodes_attributes = {'logits': {'kind': 'op'},
                    'logits_data': {'shape': None, 'value': None, 'kind': 'data'},
                    'seq_mask': {'kind': 'op'},
                    'seq_mask_data': {'shape': None, 'value': None, 'kind': 'data'},
                    'ctcgreedydecoder_node': {'op': 'CTCGreedyDecoder', 'kind': 'op',
                                              'ctc_merge_repeated': True},
                    'output': {'shape': None, 'value': None, 'kind': 'data'}}

# graph 1
edges1 = [('logits', 'logits_data'),
          ('seq_mask', 'seq_mask_data'),
          ('logits_data', 'ctcgreedydecoder_node', {'in': 0}),
          ('seq_mask_data', 'ctcgreedydecoder_node', {'in': 1}),
          ('ctcgreedydecoder_node', 'output', {'out': 0})]

# valid test case
inputs1 = {'logits_data': {'shape': int64_array([100, 4, 5])},
           'seq_mask_data': {'shape': int64_array([100, 4])}}

# invalid test case with incorrect rank for the first input tensor
inputs1_inv = {'logits_data': {'shape': int64_array([100, 4, 5, 6])},
               'seq_mask_data': {'shape': int64_array([100, 4])}}

# invalid test case with incorrect rank for the second input tensor
inputs2_inv = {'logits_data': {'shape': int64_array([100, 4, 5])},
               'seq_mask_data': {'shape': int64_array([100])}}

# invalid test case with incorrect time dimension
inputs3_inv = {'logits_data': {'shape': int64_array([100, 4, 5])},
               'seq_mask_data': {'shape': int64_array([101, 4])}}

# invalid test case with incorrect batch dimension
inputs4_inv = {'logits_data': {'shape': int64_array([100, 4, 5])},
               'seq_mask_data': {'shape': int64_array([100, 14])}}

class TestCTCGreedyDecoder(unittest.TestCase):
    def test_infer1(self):
        graph = build_graph(nodes_attributes, edges1, inputs1)
        ctcgreedydecoder_node = Node(graph, 'ctcgreedydecoder_node')
        CTCGreedyDecoderOp.infer(ctcgreedydecoder_node)

        # prepare reference results
        ref_output_shape = int64_array([4, 100, 1, 1])

        # get the result
        res_output_shape = graph.node['output']['shape']

        self.assertTrue(np.array_equal(ref_output_shape, res_output_shape),
                        'shapes do not match expected: {} and given: {}'.format(ref_output_shape, res_output_shape))

    def test_infer_invalid1(self):
        graph = build_graph(nodes_attributes, edges1, inputs1_inv)
        ctcgreedydecoder_node = Node(graph, 'ctcgreedydecoder_node')
        self.assertRaises(AssertionError, CTCGreedyDecoderOp.infer, ctcgreedydecoder_node)

    def test_infer_invalid2(self):
        graph = build_graph(nodes_attributes, edges1, inputs2_inv)
        ctcgreedydecoder_node = Node(graph, 'ctcgreedydecoder_node')
        self.assertRaises(AssertionError, CTCGreedyDecoderOp.infer, ctcgreedydecoder_node)

    def test_infer_invalid3(self):
        graph = build_graph(nodes_attributes, edges1, inputs3_inv)
        ctcgreedydecoder_node = Node(graph, 'ctcgreedydecoder_node')
        self.assertRaises(AssertionError, CTCGreedyDecoderOp.infer, ctcgreedydecoder_node)

    def test_infer_invalid4(self):
        graph = build_graph(nodes_attributes, edges1, inputs4_inv)
        ctcgreedydecoder_node = Node(graph, 'ctcgreedydecoder_node')
        self.assertRaises(AssertionError, CTCGreedyDecoderOp.infer, ctcgreedydecoder_node)
