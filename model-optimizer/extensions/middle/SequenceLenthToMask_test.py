# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from extensions.middle.SequenceLengthToMask import SequenceLengthToMask
from mo.front.common.partial_infer.utils import int64_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph


nodes_attributes = {'logits': {'shape': int64_array([5, 3, 30]), 'type': 'Parameter', 'kind': 'op',
                               'op': 'Parameter'},
                    'logits_data': {'value': None, 'shape': int64_array([5, 3, 30]), 'kind': 'data'},
                    'seq_length': {'type': 'Const', 'kind': 'op', 'op': 'Const', 'value': int64_array([5, 2, 3])},
                    'seq_length_data': {'value': int64_array([5, 2, 3]), 'kind': 'data'},
                    'ctc_greedy_decoder': {'type': None, 'kind': 'op', 'op': 'CTCGreedyDecoder',
                                           'use_mask_format': True},
                    'ctc_greedy_decoder_data': {'value': None, 'shape': None, 'kind': 'data'},
                    'last': {'kind': 'op', 'op': 'Result'},

                    # new nodes
                    'seq_mask': {'type': 'Const', 'kind': 'op', 'op': 'Const',
                                 'value': np.transpose(np.array([[1, 1, 1, 1, 1],
                                                                 [1, 1, 0, 0, 0],
                                                                 [1, 1, 1, 0, 0]], dtype=np.float))},
                    'seq_mask_data': {'value': None, 'kind': 'data'},
                    'new_ctc_greedy_decoder': {'type': None, 'kind': 'op', 'op': 'CTCGreedyDecoder'},
                    }

class ScaleInputTests(unittest.TestCase):
    def test1(self):
        graph = build_graph(nodes_attributes,
                            [('logits', 'logits_data'),
                             ('logits_data', 'ctc_greedy_decoder'),
                             ('seq_length', 'seq_length_data'),
                             ('seq_length_data', 'ctc_greedy_decoder'),
                             ('ctc_greedy_decoder', 'ctc_greedy_decoder_data'),
                             ('ctc_greedy_decoder_data', 'last')],
                            nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('logits', 'logits_data'),
                                 ('logits_data', 'new_ctc_greedy_decoder'),
                                 ('seq_mask', 'seq_mask_data'),
                                 ('seq_mask_data', 'new_ctc_greedy_decoder'),
                                 ('new_ctc_greedy_decoder', 'ctc_greedy_decoder_data'),
                                 ('ctc_greedy_decoder_data', 'last')],
                                nodes_with_edges_only=True)
        SequenceLengthToMask().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, 'last')
        self.assertTrue(flag, resp)
