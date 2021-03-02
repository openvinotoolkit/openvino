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

import numpy as np
import unittest
from argparse import Namespace

from extensions.front.tf.CTCLossReplacement import CTCLossReplacement
from mo.front.common.partial_infer.utils import int64_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph, const


class CTCLossFrontReplacementTest(unittest.TestCase):
    def test1(self):
        nodes_attributes = {
            'logits': {'shape': int64_array([2, 6, 100]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
            'seq_mask': {'shape': int64_array([2]), 'data_type': np.int32, 'kind': 'op', 'op': 'Parameter'},
            'transpose': {'kind': 'op', 'op': 'Transpose'},
            'ctc_greedy_decoder': {'kind': 'op', 'op': 'CTCGreedyDecoderSeqLen', 'merge_repeated': True},
            'cast': {'kind': 'op', 'op': 'Cast'},
            'sparse_to_dense': {'kind': 'op', 'op': 'SparseToDense'},
            'tf_ctc_loss': {'kind': 'op', 'op': 'CTCLoss', 'preprocess_collapse_repeated': False,
                         'ctc_merge_repeated': True, 'unique': False, 'logits_time_major': True},
            'ctc_loss': {'kind': 'op', 'op': 'CTCLoss', 'preprocess_collapse_repeated': False,
                            'ctc_merge_repeated': True, 'unique': False},
            **const('default_value', int64_array(-1)),
            'last': {'type': None, 'value': None, 'kind': 'op', 'op': 'Result'},
            'transpose2': {'kind': 'op', 'op': 'Transpose'},
            **const('transpose2_axis', int64_array([1, 0, 2])),
        }
        graph = build_graph(nodes_attributes, [('logits', 'transpose', {'out': 0, 'in': 0}),
                                               ('transpose', 'ctc_greedy_decoder', {'out': 0, 'in': 0}),
                                               ('seq_mask', 'ctc_greedy_decoder', {'out': 0, 'in': 1}),
                                               ('transpose', 'tf_ctc_loss', {'out': 0, 'in': 0}),
                                               ('seq_mask', 'tf_ctc_loss', {'out': 0, 'in': 3}),
                                               ('ctc_greedy_decoder', 'sparse_to_dense', {'out': 0, 'in': 0}),
                                               ('ctc_greedy_decoder', 'sparse_to_dense', {'out': 2, 'in': 1}),
                                               ('ctc_greedy_decoder', 'sparse_to_dense', {'out': 1, 'in': 2}),
                                               ('default_value', 'sparse_to_dense', {'out': 0, 'in': 3}),
                                               ('ctc_greedy_decoder', 'cast', {'out': 1, 'in': 0}),
                                               ('ctc_greedy_decoder', 'tf_ctc_loss', {'out': 0, 'in': 1}),
                                               ('cast', 'tf_ctc_loss', {'out': 0, 'in': 2}),
                                               ('tf_ctc_loss', 'last', {'out': 0, 'in': 0})],
                                               nodes_with_edges_only=True)
        graph.graph['cmd_params'] = Namespace(data_type='FP32')
        graph.stage = 'front'
        CTCLossReplacement().find_and_replace_pattern(graph)

        graph_ref = build_graph(nodes_attributes,
                                [('logits', 'transpose', {'out': 0, 'in': 0}),
                                 ('transpose', 'transpose2', {'out': 0, 'in': 0}),
                                 ('transpose2_axis', 'transpose2', {'out': 0, 'in': 1}),
                                 ('transpose2', 'ctc_greedy_decoder', {'out': 0, 'in': 0}),
                                 ('seq_mask', 'ctc_greedy_decoder', {'out': 0, 'in': 1}),
                                 ('transpose2', 'ctc_loss', {'out': 0, 'in': 0}),
                                 ('ctc_greedy_decoder', 'ctc_loss', {'out': 0, 'in': 2}),
                                 ('ctc_greedy_decoder', 'ctc_loss', {'out': 1, 'in': 3}),
                                 ('seq_mask', 'ctc_loss', {'out': 0, 'in': 1}),
                                 ('ctc_loss', 'last', {'out': 0, 'in': 0})],
                                 nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'last', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test2(self):
        nodes_attributes = {
            'logits': {'shape': int64_array([2, 6, 100]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
            'seq_mask': {'shape': int64_array([2]), 'data_type': np.int32, 'kind': 'op', 'op': 'Parameter'},
            'transpose': {'kind': 'op', 'op': 'Transpose'},
            'ctc_greedy_decoder': {'kind': 'op', 'op': 'CTCGreedyDecoderSeqLen', 'merge_repeated': True},
            'cast': {'kind': 'op', 'op': 'Cast'},
            'sparse_to_dense': {'kind': 'op', 'op': 'SparseToDense'},
            'tf_ctc_loss': {'kind': 'op', 'op': 'CTCLoss', 'preprocess_collapse_repeated': False,
                         'ctc_merge_repeated': True, 'unique': False, 'logits_time_major': False},
            'ctc_loss': {'kind': 'op', 'op': 'CTCLoss', 'preprocess_collapse_repeated': False,
                         'ctc_merge_repeated': True, 'unique': False},
            **const('default_value', int64_array(-1)),
            'last': {'type': None, 'value': None, 'kind': 'op', 'op': 'Result'},
            'transpose2': {'kind': 'op', 'op': 'Transpose'},
            **const('transpose2_axis', int64_array([1, 0, 2])),
        }
        graph = build_graph(nodes_attributes, [('logits', 'transpose', {'out': 0, 'in': 0}),
                                               ('transpose', 'ctc_greedy_decoder', {'out': 0, 'in': 0}),
                                               ('seq_mask', 'ctc_greedy_decoder', {'out': 0, 'in': 1}),
                                               ('transpose', 'tf_ctc_loss', {'out': 0, 'in': 0}),
                                               ('seq_mask', 'tf_ctc_loss', {'out': 0, 'in': 3}),
                                               ('ctc_greedy_decoder', 'sparse_to_dense', {'out': 0, 'in': 0}),
                                               ('ctc_greedy_decoder', 'sparse_to_dense', {'out': 2, 'in': 1}),
                                               ('ctc_greedy_decoder', 'sparse_to_dense', {'out': 1, 'in': 2}),
                                               ('default_value', 'sparse_to_dense', {'out': 0, 'in': 3}),
                                               ('ctc_greedy_decoder', 'cast', {'out': 1, 'in': 0}),
                                               ('ctc_greedy_decoder', 'tf_ctc_loss', {'out': 0, 'in': 1}),
                                               ('cast', 'tf_ctc_loss', {'out': 0, 'in': 2}),
                                               ('tf_ctc_loss', 'last', {'out': 0, 'in': 0})],
                                               nodes_with_edges_only=True)
        graph.graph['cmd_params'] = Namespace(data_type='FP32')
        graph.stage = 'front'
        CTCLossReplacement().find_and_replace_pattern(graph)

        graph_ref = build_graph(nodes_attributes,
                                [('logits', 'transpose', {'out': 0, 'in': 0}),
                                 ('transpose', 'transpose2', {'out': 0, 'in': 0}),
                                 ('transpose2_axis', 'transpose2', {'out': 0, 'in': 1}),
                                 ('transpose2', 'ctc_greedy_decoder', {'out': 0, 'in': 0}),
                                 ('seq_mask', 'ctc_greedy_decoder', {'out': 0, 'in': 1}),
                                 ('transpose', 'ctc_loss', {'out': 0, 'in': 0}),
                                 ('ctc_greedy_decoder', 'ctc_loss', {'out': 0, 'in': 2}),
                                 ('ctc_greedy_decoder', 'ctc_loss', {'out': 1, 'in': 3}),
                                 ('seq_mask', 'ctc_loss', {'out': 0, 'in': 1}),
                                 ('ctc_loss', 'last', {'out': 0, 'in': 0})],
                                 nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'last', check_op_attrs=True)
        self.assertTrue(flag, resp)
