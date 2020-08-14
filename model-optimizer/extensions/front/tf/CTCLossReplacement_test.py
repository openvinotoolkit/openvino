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
            'seq_mask': {'shape': int64_array([2, 100]), 'data_type': np.int32, 'kind': 'op', 'op': 'Parameter'},

            'reduce_seq_mask': {'kind': 'op', 'op': 'ReduceSum'},
            's_cast_seq_mask': {'kind': 'op', 'op': 'Cast'},
            'transpose_cast_seq_mask': {'kind': 'op', 'op': 'Transpose'},

            'transpose': {'kind': 'op', 'op': 'Transpose'},
            'ctc_greedy_decoder': {'kind': 'op', 'op': 'CTCGreedyDecoder'},
            'cast': {'kind': 'op', 'op': 'Cast'},
            'sparse_to_dense': {'kind': 'op', 'op': 'SparseToDense'},
            'const': {'kind': 'op', 'op': 'Const'},
            'ctc_loss': {'kind': 'op', 'op': 'CTCLoss', 'preprocess_collapse_repeated': False,
                         'ctc_merge_repeated': True, 'unique': False},

            'equal_op': {'kind': 'op', 'op': 'Equal'},

            'ctc_greedy_decoder_op': {'kind': 'op', 'op': 'CTCGreedyDecoder'},
            'ctc_loss_op': {'kind': 'op', 'op': 'CTCLoss'},
            'squeeze_op': {'kind': 'op', 'op': 'Squeeze'},

            'cast_labels_op': {'kind': 'op', 'op': 'Cast', 'type': 'Convert'},
            'labels_shape_op': {'kind': 'op', 'op': 'ShapeOf'},
            'broadcast_one_op': {'kind': 'op', 'op': 'Broadcast'},
            'broadcast_zero_op': {'kind': 'op', 'op': 'Broadcast'},
            'select_op': {'kind': 'op', 'op': 'Select'},
            'label_length_op': {'kind': 'op', 'op': 'ReduceSum'},

            **const('reduce_indices', int64_array(1)),
            **const('permute_order', int64_array([1, 0])),
            **const('default_value', int64_array(-1)),
            **const('squeeze_axis', int64_array([2, 3])),
            **const('minus_one', np.array([-1], dtype=np.int32)),
            **const('one', np.array([1], dtype=np.int32)),
            **const('zero', np.array([0], dtype=np.int32)),
            **const('reduce_sum_axis', int64_array([1])),

            'last': {'type': None, 'value': None, 'kind': 'op', 'op': 'Result'},
        }

        graph = build_graph(nodes_attributes,
                            [('logits', 'transpose', {'out': 0, 'in': 0}),
                             ('transpose', 'ctc_greedy_decoder', {'out': 0, 'in': 0}),
                             ('seq_mask', 'ctc_greedy_decoder', {'out': 0, 'in': 1}),

                             ('transpose', 'ctc_loss', {'out': 0, 'in': 0}),
                             ('seq_mask', 'ctc_loss', {'out': 0, 'in': 3}),

                             ('ctc_greedy_decoder', 'sparse_to_dense', {'out': 0, 'in': 0}),
                             ('ctc_greedy_decoder', 'sparse_to_dense', {'out': 2, 'in': 1}),
                             ('ctc_greedy_decoder', 'sparse_to_dense', {'out': 1, 'in': 2}),
                             ('default_value', 'sparse_to_dense', {'out': 0, 'in': 3}),
                             ('ctc_greedy_decoder', 'cast', {'out': 1, 'in': 0}),
                             ('ctc_greedy_decoder', 'ctc_loss', {'out': 0, 'in': 1}),
                             ('cast', 'ctc_loss', {'out': 0, 'in': 2}),

                             ('ctc_loss', 'last', {'out': 0, 'in': 0}),
                             ], nodes_with_edges_only=True)
        graph.graph['cmd_params'] = Namespace(data_type='FP32')
        graph.stage = 'front'
        CTCLossReplacement().find_and_replace_pattern(graph)

        graph_ref = build_graph(nodes_attributes,
                                [('seq_mask', 'reduce_seq_mask', {'out': 0, 'in': 0}),
                                 ('reduce_indices', 'reduce_seq_mask', {'out': 0, 'in': 1}),

                                 ('seq_mask', 's_cast_seq_mask', {'out': 0, 'in': 0}),
                                 ('s_cast_seq_mask', 'transpose_cast_seq_mask', {'out': 0, 'in': 0}),
                                 ('permute_order', 'transpose_cast_seq_mask', {'out': 0, 'in': 1}),

                                 ('logits', 'transpose', {'out': 0, 'in': 0}),
                                 ('transpose', 'ctc_greedy_decoder_op', {'out': 0, 'in': 0}),
                                 ('transpose_cast_seq_mask', 'ctc_greedy_decoder_op', {'out': 0, 'in': 1}),

                                 ('ctc_greedy_decoder_op', 'squeeze_op', {'out': 0, 'in': 0}),
                                 ('squeeze_axis', 'squeeze_op', {'out': 0, 'in': 1}),
                                 ('squeeze_op', 'cast_labels_op', {'in': 0}),

                                 ('minus_one', 'equal_op', {'out': 0, 'in': 1}),

                                 ('equal_op', 'labels_shape_op', {'out': 0, 'in': 0}),
                                 ('one', 'broadcast_one_op', {'out': 0, 'in': 0}),
                                 ('labels_shape_op', 'broadcast_one_op', {'out': 0, 'in': 1}),
                                 ('zero', 'broadcast_zero_op', {'out': 0, 'in': 0}),
                                 ('labels_shape_op', 'broadcast_zero_op', {'out': 0, 'in': 1}),

                                 ('equal_op', 'select_op', {'out': 0, 'in': 0}),
                                 ('broadcast_zero_op', 'select_op', {'out': 0, 'in': 1}),
                                 ('broadcast_one_op', 'select_op', {'out': 0, 'in': 2}),

                                 ('select_op', 'label_length_op', {'out': 0, 'in': 0}),
                                 ('reduce_sum_axis', 'label_length_op', {'out': 0, 'in': 1}),

                                 ('logits', 'ctc_loss_op', {'out': 0, 'in': 0}),
                                 ('reduce_seq_mask', 'ctc_loss_op', {'out': 0, 'in': 1}),
                                 ('cast_labels_op', 'ctc_loss_op', {'out': 0, 'in': 2}),
                                 ('label_length_op', 'ctc_loss_op', {'out': 0, 'in': 3}),

                                 ('cast_labels_op', 'equal_op', {'out': 0, 'in': 0}),

                                 ('ctc_loss_op', 'last', {'out': 0, 'in': 0})],
                                nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'last', check_op_attrs=True)
        self.assertTrue(flag, resp)
