# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.ops.ctc_greedy_decoder_seq_len import CTCGreedyDecoderSeqLenOp
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.graph.graph import Node
from unit_tests.utils.graph import build_graph


nodes_attributes = {'logits': {'kind': 'op'},
                    'logits_data': {'shape': None, 'value': None, 'kind': 'data'},
                    'seq_mask': {'kind': 'op'},
                    'seq_mask_data': {'shape': None, 'value': None, 'kind': 'data'},
                    'ctcgreedydecoder_node': {'op': 'CTCGreedyDecoderSeqLen', 'kind': 'op',
                                              'ctc_merge_repeated': True},
                    'output1': {'shape': None, 'value': None, 'kind': 'data'},
                    'last_output1': {'shape': None, 'value': None, 'kind': 'op'},
                    'output2': {'shape': None, 'value': None, 'kind': 'data'}
                    }

# graph 1
edges1 = [('logits', 'logits_data'),
          ('seq_mask', 'seq_mask_data'),
          ('logits_data', 'ctcgreedydecoder_node', {'in': 0}),
          ('seq_mask_data', 'ctcgreedydecoder_node', {'in': 1}),
          ('ctcgreedydecoder_node', 'output1', {'out': 0}),
          ('ctcgreedydecoder_node', 'output2', {'out': 1}),
          ('output1', 'last_output1', {'out': 0}),]

# valid test case
inputs1 = {'logits_data': {'shape': int64_array([4, 100, 5])},
           'seq_mask_data': {'shape': int64_array([4])}}

# invalid test case with incorrect rank for the first input tensor
inputs1_inv = {'logits_data': {'shape': int64_array([4, 100, 5, 6])},
               'seq_mask_data': {'shape': int64_array([4])}}

# invalid test case with incorrect rank for the second input tensor
inputs2_inv = {'logits_data': {'shape': int64_array([4, 100, 5])},
               'seq_mask_data': {'shape': int64_array([4, 100])}}

# invalid test case with incorrect time dimension
inputs3_inv = {'logits_data': {'shape': int64_array([4, 100, 5])},
               'seq_mask_data': {'shape': int64_array([4, 101])}}

# invalid test case with incorrect batch dimension
inputs4_inv = {'logits_data': {'shape': int64_array([4, 100, 5])},
               'seq_mask_data': {'shape': int64_array([14, 100])}}

class TestCTCGreedyDecoder(unittest.TestCase):
    def test_infer1(self):
        graph = build_graph(nodes_attributes, edges1, inputs1)
        ctcgreedydecoder_node = Node(graph, 'ctcgreedydecoder_node')
        CTCGreedyDecoderSeqLenOp.infer(ctcgreedydecoder_node)

        # prepare reference results
        ref_output1_shape = int64_array([4, 100])

        # get the result
        res_output1_shape = graph.node['output1']['shape']

        self.assertTrue(np.array_equal(ref_output1_shape, res_output1_shape),
                        'shapes do not match expected: {} and given: {}'.format(ref_output1_shape, res_output1_shape))

    def test_infer_invalid1(self):
        graph = build_graph(nodes_attributes, edges1, inputs1_inv)
        ctcgreedydecoder_node = Node(graph, 'ctcgreedydecoder_node')
        self.assertRaises(AssertionError, CTCGreedyDecoderSeqLenOp.infer, ctcgreedydecoder_node)

    def test_infer_invalid2(self):
        graph = build_graph(nodes_attributes, edges1, inputs2_inv)
        ctcgreedydecoder_node = Node(graph, 'ctcgreedydecoder_node')
        self.assertRaises(AssertionError, CTCGreedyDecoderSeqLenOp.infer, ctcgreedydecoder_node)

    def test_infer_invalid3(self):
        graph = build_graph(nodes_attributes, edges1, inputs3_inv)
        ctcgreedydecoder_node = Node(graph, 'ctcgreedydecoder_node')
        self.assertRaises(AssertionError, CTCGreedyDecoderSeqLenOp.infer, ctcgreedydecoder_node)

    def test_infer_invalid4(self):
        graph = build_graph(nodes_attributes, edges1, inputs4_inv)
        ctcgreedydecoder_node = Node(graph, 'ctcgreedydecoder_node')
        self.assertRaises(AssertionError, CTCGreedyDecoderSeqLenOp.infer, ctcgreedydecoder_node)
