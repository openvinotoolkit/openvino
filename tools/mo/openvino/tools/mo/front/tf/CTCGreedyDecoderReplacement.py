# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

from openvino.tools.mo.ops.ctc_greedy_decoder_seq_len import CTCGreedyDecoderSeqLenOp
from openvino.tools.mo.ops.transpose import Transpose
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph, FrontReplacementPattern
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Graph, rename_nodes
from openvino.tools.mo.ops.result import Result


def replace_ctc_greedy_decoder(graph: Graph, match: dict):
    ctc_greedy_decoder_tf = match['decoder']
    cast = match['cast']
    sparse_to_dense = match['sparse_to_dense']
    sparse_to_dense_name = sparse_to_dense.soft_get('name', sparse_to_dense.id)
    ctc_greedy_decoder_tf_name = ctc_greedy_decoder_tf.soft_get('name', ctc_greedy_decoder_tf.id)

    # For normalizing input channel needs to transpose input data from [T, N, C] to [N, T, C]
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

    # Set output of the new sub-graph as a source for SparseToDense consumer
    sparse_to_dense.out_port(0).get_connection().set_source(ctc_greedy_decoder.out_port(0))

    # Remove no longer needed nodes
    graph.remove_nodes_from([sparse_to_dense.id, cast.id, ctc_greedy_decoder_tf.id])


class CTCGreedyDecoderReplacement(FrontReplacementSubgraph):
    """
    TensorFlow CTCGreedyDecoder produces output in a sparse tensor that is not supported by OpenVINO, and
    OpenVINO's CTCGreedyDecoderSeqLen has a different output that is in a dense format. So this transformation
    intents to replace TF CTCGreedyDecoder+SparseToDense where SparseToDense third input get from input parameter
    to CTCGreedyDecoderSeqLen which compatible with IE.
    """
    enabled = True

    @staticmethod
    def pattern(**kwargs):
        return dict(
            nodes=[('decoder', dict(op='CTCGreedyDecoderSeqLen', output_sparse_format=True)),
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
    TensorFlow CTCGreedyDecoder produces output in a sparse tensor that is not supported by OpenVINO, and
    OpenVINO's CTCGreedyDecoderSeqLen has a different output that is in a dense format. So this transformation
    intents to replace TF CTCGreedyDecoder+SparseToDense where SparseToDense third input get from CTCGreedyDecoder
    second output to CTCGreedyDecoderSeqLen which compatible with IE.
    """
    enabled = True

    @staticmethod
    def pattern(**kwargs):
        return dict(
            nodes=[('decoder', dict(op='CTCGreedyDecoderSeqLen', output_sparse_format=True)),
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


class CTCGreedyDecoderSingleReplacement(FrontReplacementPattern):
    """
    TensorFlow CTCGreedyDecoder produces output in a sparse tensor that is not supported by OpenVINO, and
    OpenVINO's CTCGreedyDecoderSeqLen has a different output that is in a dense format. So this transformation
    handles a single TF CTCGreedyDecoder and warns the user about another format of the output
    """
    enabled = True

    def run_after(self):
        return [CTCGreedyDecoderReplacement, CTCGreedyDecoderWithSparseToDenseShapeReplacement]

    def find_and_replace_pattern(self, graph: Graph):
        for ctc_greedy_decoder_tf in graph.get_op_nodes(op='CTCGreedyDecoderSeqLen', output_sparse_format=True):
            ctc_greedy_decoder_tf_name = ctc_greedy_decoder_tf.soft_get('name', ctc_greedy_decoder_tf.id)

            # TF CTCGreedyDecoder have 4 output tensors. If any of them connected to not Result operation then
            # transformation in not applicable
            for port_num in ctc_greedy_decoder_tf.out_ports():
                if not ctc_greedy_decoder_tf.out_port(port_num).disconnected()\
                        and ctc_greedy_decoder_tf.out_port(port_num).get_destination().node.soft_get('op') != 'Result':
                    return

            # If the first and second output are not connected to Result operations -
            # create Result operation and connect it to appropriate output
            if ctc_greedy_decoder_tf.out_port(0).disconnected():
                first_result = Result(graph,
                                       {'name': ctc_greedy_decoder_tf_name + '/decoded_classes'}
                                       ).create_node()
                ctc_greedy_decoder_tf.out_port(0).connect(first_result.in_port(0))

            if ctc_greedy_decoder_tf.out_port(1).disconnected():
                second_result = Result(graph,
                                       {'name': ctc_greedy_decoder_tf_name + '/seq_lengths_output'}
                                       ).create_node()
                ctc_greedy_decoder_tf.out_port(1).connect(second_result.in_port(0))


            # For normalizing input channel needs to transpose input data from [T, N, C] to [N, T, C]
            # which supported CTCGreedyDecoderSeqLen op.
            log.warning('Found TF CTCGreedyDecoder operation at the end of network. '
                        'PLEASE NOTE, appropriate network output operation CTCGreedyDecoderSeqLen {} '
                        'will have dense format, not sparse format!'.format(ctc_greedy_decoder_tf_name))
            ctc_data_permute = create_op_with_const_inputs(graph, Transpose, {1: int64_array([1, 0, 2])},
                                                           {'name': ctc_greedy_decoder_tf_name + '/ctc_data_permute'})

            assert ctc_greedy_decoder_tf.has_valid('merge_repeated'), \
                'The CTCGreedyDecoderSeqLen node "{}" misses "merge_repeated" attribute'.format(
                    ctc_greedy_decoder_tf_name)

            ctc_greedy_decoder_tf.in_port(0).get_source().connect(ctc_data_permute.in_port(0))
            ctc_greedy_decoder_tf.in_port(0).disconnect()
            ctc_data_permute.out_port(0).connect(ctc_greedy_decoder_tf.in_port(0))

            del ctc_greedy_decoder_tf['output_sparse_format']

            for port_num in [2, 3]:  # MO CTCGreedyDecoderSeqLen may have 2 outputs
                if port_num in ctc_greedy_decoder_tf.out_ports():
                    if not ctc_greedy_decoder_tf.out_port(port_num).disconnected():
                        ctc_greedy_decoder_tf.out_port(port_num).disconnect()
