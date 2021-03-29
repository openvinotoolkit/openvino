# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.ctc_greedy_decoder_seq_len import CTCGreedyDecoderSeqLenOp
from extensions.ops.transpose import Transpose
from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, rename_nodes


def replace_ctc_greedy_decoder(graph: Graph, match: dict):
    ctc_greedy_decoder_tf = match['decoder']
    cast = match['cast']
    sparse_to_dense = match['sparse_to_dense']
    sparse_to_dense_name = sparse_to_dense.soft_get('name', sparse_to_dense.id)
    ctc_greedy_decoder_tf_name = ctc_greedy_decoder_tf.soft_get('name', ctc_greedy_decoder_tf.id)

    # for normalizing input chanel need to transpose input data from [T, N, C] to [N, T, C]
    # which supported CTCGreedyDecoderSeqLen op.
    ctc_data_permute = create_op_with_const_inputs(graph, Transpose, {1: int64_array([1, 0, 2])},
                                                   {'name': ctc_greedy_decoder_tf_name + '/ctc_data_permute'})

    assert ctc_greedy_decoder_tf.has_valid('merge_repeated'), \
        'The CTCGreedyDecoderSeqLen node "{}" misses "merge_repeated" attribute'.format(ctc_greedy_decoder_tf_name)

    ctc_greedy_decoder_tf.in_port(0).get_source().connect(ctc_data_permute.in_port(0))
    merge_repeated_tf = ctc_greedy_decoder_tf.merge_repeated
    ctc_greedy_decoder = CTCGreedyDecoderSeqLenOp(graph, {'name': sparse_to_dense_name,
                                                          'merge_repeated': merge_repeated_tf}).create_node()
    rename_nodes(
        [(sparse_to_dense, sparse_to_dense_name + '/AbandonedName'), (ctc_greedy_decoder, sparse_to_dense_name)])
    ctc_greedy_decoder.in_port(0).connect(ctc_data_permute.out_port(0))
    ctc_greedy_decoder_tf.in_port(1).get_source().connect(ctc_greedy_decoder.in_port(1))

    # set output of the new sub-graph as a source for SparseToDense consumer
    sparse_to_dense.out_port(0).get_connection().set_source(ctc_greedy_decoder.out_port(0))

    # remove no longer needed nodes
    graph.remove_nodes_from([sparse_to_dense.id, cast.id, ctc_greedy_decoder_tf.id])


class CTCGreedyDecoderReplacement(FrontReplacementSubgraph):
    """
    TensorFlow CTCGreedyDecoder produces output in a sparse tensor that is not supported by Inference Engine and
    Inference Engine's CTCGreedyDecoderSeqLen has different output that is in a dense format. So this transformation
    intents to replace TF CTCGreedyDecoder+SparseToDense where SparseToDense third input get from input parameter
    to CTCGreedyDecoderSeqLen which compatible with IE.
    """
    enabled = True

    @staticmethod
    def pattern(**kwargs):
        return dict(
            nodes=[('decoder', dict(op='CTCGreedyDecoderSeqLen')),
                   ('cast', dict(op='Cast')),
                   ('sparse_to_dense', dict(op='SparseToDense'))
                   ],
            edges=[('decoder', 'sparse_to_dense', {'out': 0}),
                   ('decoder', 'cast', {'out': 1}),
                   ('cast', 'sparse_to_dense', {'out': 0})]
        )

    def replace_sub_graph(self, graph: Graph, match: dict):
        replace_ctc_greedy_decoder(graph, match)


class CTCGreedyDecoderWithSparseToDenseShapeReplacement(FrontReplacementSubgraph):
    """
    TensorFlow CTCGreedyDecoder produces output in a sparse tensor that is not supported by Inference Engine and
    Inference Engine's CTCGreedyDecoderSeqLen has different output that is in a dense format. So this transformation
    intents to replace TF CTCGreedyDecoder+SparseToDense where SparseToDense third input get from CTCGreedyDecoder
    second output to CTCGreedyDecoderSeqLen which compatible with IE.
    """
    enabled = True

    @staticmethod
    def pattern(**kwargs):
        return dict(
            nodes=[('decoder', dict(op='CTCGreedyDecoderSeqLen')),
                   ('cast', dict(op='Cast')),
                   ('sparse_to_dense', dict(op='SparseToDense'))
                   ],
            edges=[('decoder', 'sparse_to_dense', {'out': 0}),
                   ('decoder', 'cast', {'out': 1}),
                   ('decoder', 'sparse_to_dense', {'out': 2}),
                   ('cast', 'sparse_to_dense', {'out': 0})]
        )

    def replace_sub_graph(self, graph: Graph, match: dict):
        replace_ctc_greedy_decoder(graph, match)
