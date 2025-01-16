# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.front.tf.CTCGreedyDecoderReplacement import CTCGreedyDecoderReplacement, \
    CTCGreedyDecoderWithSparseToDenseShapeReplacement, CTCGreedyDecoderSingleReplacement
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, const


class CTCGreedyDecoderReplacementTests(unittest.TestCase):
    nodes_attributes = {
        # nodes from original graph
        'logits': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
        'seq_len': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
        'order_arr': {'kind': 'op', 'op': 'Const'},
        'transpose': {'type': 'Transpose', 'kind': 'op', 'op': 'Transpose'},
        'decoder': {'kind': 'op', 'op': 'CTCGreedyDecoderSeqLen', 'merge_repeated': True, 'output_sparse_format': True},
        'cast': {'kind': 'op', 'op': 'Cast'},
        'sparse_to_dense': {'kind': 'op', 'op': 'SparseToDense'},
        'last': {'type': None, 'value': None, 'kind': 'op', 'op': 'Result'},
        'last_1': {'type': None, 'value': None, 'kind': 'op', 'op': 'Result'},

        # new nodes
        'new_decoder': {'kind': 'op', 'op': 'CTCGreedyDecoderSeqLen', 'merge_repeated': True},
        **const('squeeze_axes', int64_array([2, 3])),
        'squeeze_dec_seq': {'kind': 'op', 'op': 'Squeeze'},
        'cast_to_int': {'kind': 'op', 'op': 'Cast'},
        'out_seq_len': {'type': None, 'value': None, 'kind': 'op', 'op': 'Result'},
    }

    def test_CTCGreedyDecoderWithSparseToDenseShape(self):
        graph = build_graph(self.nodes_attributes,
                            [('logits', 'decoder', {'out': 0, 'in': 0}),
                             ('seq_len', 'decoder', {'out': 0, 'in': 1}),
                             ('decoder', 'sparse_to_dense', {'out': 0, 'in': 0}),
                             ('decoder', 'sparse_to_dense', {'out': 2, 'in': 1}),
                             ('decoder', 'cast', {'out': 1, 'in': 0}),
                             ('cast', 'sparse_to_dense', {'out': 0}),
                             ('sparse_to_dense', 'last', {'out': 0, 'in': 0}),
                             ], nodes_with_edges_only=True)
        graph.stage = 'front'
        CTCGreedyDecoderWithSparseToDenseShapeReplacement().find_and_replace_pattern(graph)

        graph_ref = build_graph(self.nodes_attributes,
                                [('logits', 'transpose', {'out': 0, 'in': 0}),
                                 ('order_arr', 'transpose', {'out': 0, 'in': 1}),
                                 ('transpose', 'new_decoder', {'out': 0, 'in': 0}),
                                 ('seq_len', 'new_decoder', {'out': 0, 'in': 1}),
                                 ('new_decoder', 'last', {'out': 0, 'in': 0}),
                                 ],
                                nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'last', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_CTCGreedyDecoderReplacement(self):
        graph = build_graph(self.nodes_attributes,
                            [('logits', 'decoder', {'out': 0, 'in': 0}),
                             ('seq_len', 'decoder', {'out': 0, 'in': 1}),
                             ('decoder', 'sparse_to_dense', {'out': 0, 'in': 0}),
                             ('decoder', 'cast', {'out': 1, 'in': 0}),
                             ('cast', 'sparse_to_dense', {'out': 0}),
                             ('sparse_to_dense', 'last', {'out': 0, 'in': 0}),
                             ], nodes_with_edges_only=True)
        graph.stage = 'front'
        CTCGreedyDecoderReplacement().find_and_replace_pattern(graph)

        graph_ref = build_graph(self.nodes_attributes,
                                [('logits', 'transpose', {'out': 0, 'in': 0}),
                                 ('order_arr', 'transpose', {'out': 0, 'in': 1}),
                                 ('transpose', 'new_decoder', {'out': 0, 'in': 0}),
                                 ('seq_len', 'new_decoder', {'out': 0, 'in': 1}),
                                 ('new_decoder', 'last', {'out': 0, 'in': 0}),
                                 ],
                                nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'last', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_CTCGreedyDecoderSingle(self):
        graph = build_graph(self.nodes_attributes,
                            [('logits', 'decoder', {'out': 0, 'in': 0}),
                             ('seq_len', 'decoder', {'out': 0, 'in': 1}),
                             ('decoder', 'last', {'out': 0, 'in': 0}),
                             ('decoder', 'last_1', {'out': 1, 'in': 0}),
                             ], nodes_with_edges_only=True)
        graph.stage = 'front'
        CTCGreedyDecoderSingleReplacement().find_and_replace_pattern(graph)

        graph_ref = build_graph(self.nodes_attributes,
                                [('logits', 'transpose', {'out': 0, 'in': 0}),
                                 ('order_arr', 'transpose', {'out': 0, 'in': 1}),
                                 ('transpose', 'new_decoder', {'out': 0, 'in': 0}),
                                 ('seq_len', 'new_decoder', {'out': 0, 'in': 1}),
                                 ('new_decoder', 'last', {'out': 0, 'in': 0}),
                                 ('new_decoder', 'last_1', {'out': 1, 'in': 0}),
                                 ],
                                nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'last', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_CTCGreedyDecoderSingle_negative(self):
        edges = [('logits', 'decoder', {'out': 0, 'in': 0}),
                 ('seq_len', 'decoder', {'out': 0, 'in': 1}),
                 ('decoder', 'sparse_to_dense', {'out': 0, 'in': 0}),
                 ('decoder', 'cast', {'out': 1, 'in': 0}),
                 ('cast', 'sparse_to_dense', {'out': 0}),
                 ('sparse_to_dense', 'last', {'out': 0, 'in': 0}),
                 ]
        graph = build_graph(self.nodes_attributes,
                            edges, nodes_with_edges_only=True)
        graph.stage = 'front'
        CTCGreedyDecoderSingleReplacement().find_and_replace_pattern(graph)

        graph_ref = build_graph(self.nodes_attributes,
                                edges, nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'last', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_CTCGreedyDecoder_no_consequent_transforms(self):
        graph = build_graph(self.nodes_attributes,
                            [('logits', 'decoder', {'out': 0, 'in': 0}),
                             ('seq_len', 'decoder', {'out': 0, 'in': 1}),
                             ('decoder', 'sparse_to_dense', {'out': 0, 'in': 0}),
                             ('decoder', 'sparse_to_dense', {'out': 2, 'in': 1}),
                             ('decoder', 'cast', {'out': 1, 'in': 0}),
                             ('cast', 'sparse_to_dense', {'out': 0}),
                             ('sparse_to_dense', 'last', {'out': 0, 'in': 0}),
                             ], nodes_with_edges_only=True)
        graph.stage = 'front'
        CTCGreedyDecoderWithSparseToDenseShapeReplacement().find_and_replace_pattern(graph)
        CTCGreedyDecoderSingleReplacement().find_and_replace_pattern(graph)

        graph_ref = build_graph(self.nodes_attributes,
                                [('logits', 'transpose', {'out': 0, 'in': 0}),
                                 ('order_arr', 'transpose', {'out': 0, 'in': 1}),
                                 ('transpose', 'new_decoder', {'out': 0, 'in': 0}),
                                 ('seq_len', 'new_decoder', {'out': 0, 'in': 1}),
                                 ('new_decoder', 'last', {'out': 0, 'in': 0}),
                                 ],
                                nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'last', check_op_attrs=True)
        self.assertTrue(flag, resp)
