# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.ctc_greedy_decoder_seq_len import CTCGreedyDecoderSeqLenOp
from openvino.tools.mo.ops.ctc_loss import CTCLoss
from openvino.tools.mo.ops.transpose import Transpose
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Graph, rename_nodes


class CTCLossReplacement(FrontReplacementSubgraph):
    """
    The CTCLoss appears along with CTCGreedyDecoder operation in particular. Since the TensorFlow* CTCGreedyDecoder
    outputs sparse tensor format, the OpenVINO CTCGreedyDecoderSeqLen has a different format and the CTCLoss is also affected
    in terms of different format for its inputs. So the corresponding sub-graph with CTCGreedyDecoding and CTCLoss
    must be transformed properly.
    """
    enabled = True

    def run_before(self):
        from openvino.tools.mo.front.tf.CTCGreedyDecoderReplacement import CTCGreedyDecoderReplacement
        return [CTCGreedyDecoderReplacement]

    def pattern(self):
        return dict(
            nodes=[
                ('transpose', dict(op='Transpose')),
                ('ctc_greedy_decoder', dict(op='CTCGreedyDecoderSeqLen', output_sparse_format=True)),
                ('cast', dict(op='Cast')),
                ('sparse_to_dense', dict(op='SparseToDense')),
                ('const', dict(op='Const')),
                ('ctc_loss', dict(op='CTCLoss')),
            ],
            edges=[
                ('transpose', 'ctc_greedy_decoder', {'out': 0, 'in': 0}),
                ('transpose', 'ctc_loss', {'out': 0, 'in': 0}),
                ('ctc_greedy_decoder', 'sparse_to_dense', {'out': 0, 'in': 0}),
                ('ctc_greedy_decoder', 'sparse_to_dense', {'out': 2, 'in': 1}),
                ('ctc_greedy_decoder', 'sparse_to_dense', {'out': 1, 'in': 2}),
                ('const', 'sparse_to_dense', {'out': 0, 'in': 3}),
                ('ctc_greedy_decoder', 'cast', {'out': 1, 'in': 0}),
                ('ctc_greedy_decoder', 'ctc_loss', {'out': 0, 'in': 1}),
                ('cast', 'ctc_loss', {'out': 0, 'in': 2})
            ])

    def replace_sub_graph(self, graph: Graph, match: dict):
        transpose_tf = match['transpose']
        ctc_greedy_decoder_tf = match['ctc_greedy_decoder']
        cast_tf = match['cast']
        ctc_loss_tf = match['ctc_loss']
        sparse_to_dense_tf = match['sparse_to_dense']
        output_sparse_to_dense_name = sparse_to_dense_tf.soft_get('name', sparse_to_dense_tf.id)
        ctc_data_permute = create_op_with_const_inputs(graph, Transpose, {1: int64_array([1, 0, 2])},
                                                       {'name': ctc_greedy_decoder_tf.name + '/ctc_data_permute'})
        ctc_data_permute.in_port(0).connect(transpose_tf.out_port(0))

        ctc_greedy_decoder_tf_name = ctc_greedy_decoder_tf.soft_get('name', ctc_greedy_decoder_tf.id)
        assert ctc_greedy_decoder_tf.has_valid('merge_repeated'), \
            'The CTCGreedyDecoderSeqLen node "{}" misses "merge_repeated" attribute'.format(ctc_greedy_decoder_tf_name)
        merge_repeated_tf = ctc_greedy_decoder_tf.merge_repeated
        ctc_greedy_decoder = CTCGreedyDecoderSeqLenOp(graph, {'name': output_sparse_to_dense_name,
                                                              'merge_repeated': merge_repeated_tf}).create_node()
        rename_nodes([(sparse_to_dense_tf, output_sparse_to_dense_name + '/AbandonedName'),
                      (ctc_greedy_decoder, output_sparse_to_dense_name)])
        ctc_greedy_decoder.in_port(0).connect(ctc_data_permute.out_port(0))
        ctc_greedy_decoder.in_port(1).connect(ctc_greedy_decoder_tf.in_port(1).get_connection().get_source())

        # set output of the new sub-graph as a source for SparseToDense consumer
        output_ctc_loss_name = ctc_loss_tf.soft_get('name', ctc_loss_tf.id)
        assert ctc_loss_tf.has_valid('preprocess_collapse_repeated'), \
            'The CTCLoss node "{}" misses "preprocess_collapse_repeated" attribute'.format(output_ctc_loss_name)
        assert ctc_loss_tf.has_valid('ctc_merge_repeated'), \
            'The CTCLoss node "{}" misses "ctc_merge_repeated" attribute'.format(output_ctc_loss_name)
        assert ctc_loss_tf.has_valid('unique'), \
            'The CTCLoss node "{}" misses "unique" attribute'.format(output_ctc_loss_name)
        preprocess_collapse_repeated = ctc_loss_tf.preprocess_collapse_repeated
        ctc_merge_repeated = ctc_loss_tf.ctc_merge_repeated
        unique = ctc_loss_tf.unique
        ctc_loss = CTCLoss(graph, {'name': output_ctc_loss_name,
                                   'preprocess_collapse_repeated': preprocess_collapse_repeated,
                                   'ctc_merge_repeated': ctc_merge_repeated,
                                   'unique': unique}).create_node()
        rename_nodes([(ctc_loss_tf, output_ctc_loss_name + '/AbandonedName'), (ctc_loss, output_ctc_loss_name)])
        ctc_loss_tf.out_port(0).get_connection().set_source(ctc_loss.out_port(0))
        if ctc_loss_tf.logits_time_major:
            ctc_loss.in_port(0).connect(ctc_data_permute.out_port(0))
        else:
            ctc_loss.in_port(0).connect(transpose_tf.out_port(0))
        ctc_loss.in_port(1).connect(ctc_greedy_decoder_tf.in_port(1).get_connection().get_source())
        ctc_loss.in_port(2).connect(ctc_greedy_decoder.out_port(0))
        ctc_loss.in_port(3).connect(ctc_greedy_decoder.out_port(1))

        # remove no longer needed nodes
        graph.remove_nodes_from([sparse_to_dense_tf.id, cast_tf.id, ctc_loss_tf.id, ctc_greedy_decoder_tf.id])
