"""
 Copyright (C) 2021 Intel Corporation

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

from extensions.front.tf.CTCGreedyDecoderReplacement import CTCGreedyDecoderReplacement
from mo.front.common.partial_infer.utils import int64_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph, const


class CTCGreedyDecoderReplacementTests(unittest.TestCase):
    def test1(self):
        nodes_attributes = {
            # nodes from original graph
            'logits': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
            'seq_len': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
            'order_arr': {'kind': 'op', 'op': 'Const'},
            'transpose': {'type': 'Transpose', 'kind': 'op', 'op': 'Transpose'},
            'decoder': {'kind': 'op', 'op': 'CTCGreedyDecoderSeqLen', 'merge_repeated': True},
            'cast': {'kind': 'op', 'op': 'Cast'},
            'sparse_to_dense': {'kind': 'op', 'op': 'SparseToDense'},
            'last': {'type': None, 'value': None, 'kind': 'op', 'op': 'Result'},

            # new nodes
            'new_decoder': {'kind': 'op', 'op': 'CTCGreedyDecoderSeqLen', 'use_mask_format': True},
            **const('squeeze_axes', int64_array([2, 3])),
            'squeeze_dec_seq': {'kind': 'op', 'op': 'Squeeze'},
            'cast_to_int': {'kind': 'op', 'op': 'Cast'},
        }

        graph = build_graph(nodes_attributes,
                            [('logits', 'decoder', {'out': 0, 'in': 0}),
                             ('seq_len', 'decoder', {'out': 0, 'in': 1}),
                             ('decoder', 'sparse_to_dense', {'out': 0, 'in': 0}),
                             ('decoder', 'sparse_to_dense', {'out': 2, 'in': 1}),
                             ('decoder', 'cast', {'out': 1, 'in': 0}),
                             ('cast', 'sparse_to_dense', {'out': 0}),
                             ('sparse_to_dense', 'last', {'out': 0, 'in': 0}),
                             ], nodes_with_edges_only=True)
        graph.stage = 'front'
        CTCGreedyDecoderReplacement().find_and_replace_pattern(graph)

        graph_ref = build_graph(nodes_attributes,
                                [('logits', 'transpose', {'out': 0, 'in': 0}),
                                 ('order_arr', 'transpose', {'out': 0, 'in': 1}),
                                 ('transpose', 'decoder', {'out': 0, 'in': 0}),
                                 ('seq_len', 'decoder', {'out': 0, 'in': 1}),
                                 ('decoder', 'last', {'out': 0, 'in': 0}),
                                 ],
                                nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'last', check_op_attrs=True)
        self.assertTrue(flag, resp)
