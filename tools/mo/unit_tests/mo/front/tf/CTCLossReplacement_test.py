# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import unittest
from argparse import Namespace

from openvino.tools.mo.front.tf.CTCLossReplacement import CTCLossReplacement
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, const


class CTCLossFrontReplacementTest(unittest.TestCase):
    nodes_attributes = {
        'logits': {'shape': int64_array([2, 6, 100]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
        'seq_mask': {'shape': int64_array([2]), 'data_type': np.int32, 'kind': 'op', 'op': 'Parameter'},
        'transpose': {'kind': 'op', 'op': 'Transpose'},
        'ctc_greedy_decoder': {'kind': 'op', 'op': 'CTCGreedyDecoderSeqLen', 'merge_repeated': True,
                               'output_sparse_format': True},
        'cast': {'kind': 'op', 'op': 'Cast'},
        'sparse_to_dense': {'kind': 'op', 'op': 'SparseToDense'},
        'tf_ctc_loss_true_logits': {'kind': 'op', 'op': 'CTCLoss', 'preprocess_collapse_repeated': False,
                                    'ctc_merge_repeated': True, 'unique': False, 'logits_time_major': True},
        'tf_ctc_loss_false_logits': {'kind': 'op', 'op': 'CTCLoss', 'preprocess_collapse_repeated': False,
                                     'ctc_merge_repeated': True, 'unique': False, 'logits_time_major': False},
        'ctc_loss': {'kind': 'op', 'op': 'CTCLoss', 'preprocess_collapse_repeated': False,
                     'ctc_merge_repeated': True, 'unique': False},
        **const('default_value', int64_array(-1)),
        'last': {'type': None, 'value': None, 'kind': 'op', 'op': 'Result'},
        'transpose2': {'kind': 'op', 'op': 'Transpose'},
        **const('transpose2_axis', int64_array([1, 0, 2])),

        'new_ctc_greedy_decoder': {'kind': 'op', 'op': 'CTCGreedyDecoderSeqLen', 'merge_repeated': True},
    }

    def CTCLossReplacement_test_true_logits(self):
        graph = build_graph(self.nodes_attributes,
                            [('logits', 'transpose', {'out': 0, 'in': 0}),
                             ('transpose', 'ctc_greedy_decoder', {'out': 0, 'in': 0}),
                             ('seq_mask', 'ctc_greedy_decoder', {'out': 0, 'in': 1}),
                             ('transpose', 'tf_ctc_loss_true_logits', {'out': 0, 'in': 0}),
                             ('seq_mask', 'tf_ctc_loss_true_logits', {'out': 0, 'in': 3}),
                             ('ctc_greedy_decoder', 'sparse_to_dense', {'out': 0, 'in': 0}),
                             ('ctc_greedy_decoder', 'sparse_to_dense', {'out': 2, 'in': 1}),
                             ('ctc_greedy_decoder', 'sparse_to_dense', {'out': 1, 'in': 2}),
                             ('default_value', 'sparse_to_dense', {'out': 0, 'in': 3}),
                             ('ctc_greedy_decoder', 'cast', {'out': 1, 'in': 0}),
                             ('ctc_greedy_decoder', 'tf_ctc_loss_true_logits', {'out': 0, 'in': 1}),
                             ('cast', 'tf_ctc_loss_true_logits', {'out': 0, 'in': 2}),
                             ('tf_ctc_loss_true_logits', 'last', {'out': 0, 'in': 0})],
                            nodes_with_edges_only=True)
        graph.graph['cmd_params'] = Namespace(data_type='FP32')
        graph.stage = 'front'
        CTCLossReplacement().find_and_replace_pattern(graph)

        graph_ref = build_graph(self.nodes_attributes,
                                [('logits', 'transpose', {'out': 0, 'in': 0}),
                                 ('transpose', 'transpose2', {'out': 0, 'in': 0}),
                                 ('transpose2_axis', 'transpose2', {'out': 0, 'in': 1}),
                                 ('transpose2', 'new_ctc_greedy_decoder', {'out': 0, 'in': 0}),
                                 ('seq_mask', 'new_ctc_greedy_decoder', {'out': 0, 'in': 1}),
                                 ('transpose2', 'ctc_loss', {'out': 0, 'in': 0}),
                                 ('new_ctc_greedy_decoder', 'ctc_loss', {'out': 0, 'in': 2}),
                                 ('new_ctc_greedy_decoder', 'ctc_loss', {'out': 1, 'in': 3}),
                                 ('seq_mask', 'ctc_loss', {'out': 0, 'in': 1}),
                                 ('ctc_loss', 'last', {'out': 0, 'in': 0})],
                                nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'last', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def CTCLossReplacement_test_false_logits(self):
        graph = build_graph(self.nodes_attributes,
                            [('logits', 'transpose', {'out': 0, 'in': 0}),
                             ('transpose', 'ctc_greedy_decoder', {'out': 0, 'in': 0}),
                             ('seq_mask', 'ctc_greedy_decoder', {'out': 0, 'in': 1}),
                             ('transpose', 'tf_ctc_loss_false_logits', {'out': 0, 'in': 0}),
                             ('seq_mask', 'tf_ctc_loss_false_logits', {'out': 0, 'in': 3}),
                             ('ctc_greedy_decoder', 'sparse_to_dense', {'out': 0, 'in': 0}),
                             ('ctc_greedy_decoder', 'sparse_to_dense', {'out': 2, 'in': 1}),
                             ('ctc_greedy_decoder', 'sparse_to_dense', {'out': 1, 'in': 2}),
                             ('default_value', 'sparse_to_dense', {'out': 0, 'in': 3}),
                             ('ctc_greedy_decoder', 'cast', {'out': 1, 'in': 0}),
                             ('ctc_greedy_decoder', 'tf_ctc_loss_false_logits', {'out': 0, 'in': 1}),
                             ('cast', 'tf_ctc_loss_false_logits', {'out': 0, 'in': 2}),
                             ('tf_ctc_loss_false_logits', 'last', {'out': 0, 'in': 0})],
                            nodes_with_edges_only=True)
        graph.graph['cmd_params'] = Namespace(data_type='FP32')
        graph.stage = 'front'
        CTCLossReplacement().find_and_replace_pattern(graph)

        graph_ref = build_graph(self.nodes_attributes,
                                [('logits', 'transpose', {'out': 0, 'in': 0}),
                                 ('transpose', 'transpose2', {'out': 0, 'in': 0}),
                                 ('transpose2_axis', 'transpose2', {'out': 0, 'in': 1}),
                                 ('transpose2', 'new_ctc_greedy_decoder', {'out': 0, 'in': 0}),
                                 ('seq_mask', 'new_ctc_greedy_decoder', {'out': 0, 'in': 1}),
                                 ('transpose', 'ctc_loss', {'out': 0, 'in': 0}),
                                 ('new_ctc_greedy_decoder', 'ctc_loss', {'out': 0, 'in': 2}),
                                 ('new_ctc_greedy_decoder', 'ctc_loss', {'out': 1, 'in': 3}),
                                 ('seq_mask', 'ctc_loss', {'out': 0, 'in': 1}),
                                 ('ctc_loss', 'last', {'out': 0, 'in': 0})],
                                nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'last', check_op_attrs=True)
        self.assertTrue(flag, resp)
