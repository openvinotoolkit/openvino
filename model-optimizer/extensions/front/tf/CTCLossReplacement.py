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
import logging as log

from extensions.ops.Cast import Cast
from extensions.ops.ctc_greedy_decoder import CTCGreedyDecoderOp
from extensions.ops.ctc_loss import CTCLoss
from extensions.ops.elementwise import Equal
from extensions.ops.parameter import Parameter
from extensions.ops.ReduceOps import ReduceSum
from extensions.ops.select import Select
from extensions.ops.transpose import Transpose
from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, rename_nodes
from mo.middle.passes.convert_data_type import data_type_str_to_np
from mo.ops.broadcast import Broadcast
from mo.ops.shape import Shape
from mo.ops.squeeze import Squeeze
from mo.utils.error import Error


class CTCLossReplacement(FrontReplacementSubgraph):
    """
    The CTCLoss appears along with CTCGreedyDecoder operation in particular. Since the TensorFlow* CTCGreedyDecoder
    outputs sparse tensor format, the OpenVINO CTCGreedyDecoder has a different format and the CTCLoss is also affected
    in terms of different format for its inputs. So the corresponding sub-graph with CTCGreedyDecoding and CTCLoss
    must be transformed properly.
    Also, the transformation changes the input sequence length format into a mask format. For example, 1D tensor of
    sequence lengths equal to [4 2] is coded as 2D tensor [[1 1 1 1 0], [1 1 0 0 0]] with a time dimension is
    equal to 5.
    """
    enabled = True

    def run_before(self):
        from extensions.front.tf.CTCGreedyDecoderReplacement import CTCGreedyDecoderReplacement
        return [CTCGreedyDecoderReplacement]

    def pattern(self):
        return dict(
            nodes=[
                ('seq_len', dict(op='Parameter')),
                ('transpose', dict(op='Transpose')),
                ('ctc_greedy_decoder', dict(op='CTCGreedyDecoder')),
                ('cast', dict(op='Cast')),
                ('sparse_to_dense', dict(op='SparseToDense')),
                ('const', dict(op='Const')),
                ('ctc_loss', dict(op='CTCLoss')),
            ],
            edges=[
                ('seq_len', 'ctc_greedy_decoder', {'out': 0, 'in': 1}),
                ('seq_len', 'ctc_loss', {'out': 0, 'in': 3}),
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
        seq_len_tf = match['seq_len']
        transpose_tf = match['transpose']
        ctc_greedy_decoder_tf = match['ctc_greedy_decoder']
        cast_tf = match['cast']
        ctc_loss_tf = match['ctc_loss']
        sparse_to_dense_tf = match['sparse_to_dense']

        output_sparse_to_dense_name = sparse_to_dense_tf.soft_get('name', sparse_to_dense_tf.id)
        output_ctc_loss_name = ctc_loss_tf.soft_get('name', ctc_loss_tf.id)
        ctc_greedy_decoder_tf_name = ctc_greedy_decoder_tf.soft_get('name', ctc_greedy_decoder_tf.id)

        log.debug('Found CTCLossFrontReplacer pattern after {} with name {}'.format(ctc_greedy_decoder_tf.op,
                                                                                    ctc_greedy_decoder_tf.name))

        # create sequence mask node, sub-graph for transforming into sequence length and connect with consumers
        seq_len_tf_shape = seq_len_tf.soft_get('shape', None)
        if seq_len_tf_shape is None or len(seq_len_tf_shape) != 2:
            raise Error('The sequence length that is the second input to the CTCGreedyDecoder node "{}"'
                        ' must be specified in a mask format.'.format(ctc_greedy_decoder_tf_name))
        log.error('The format of input sequence length has been changed to a mask format', extra={'is_warning': True})
        seq_len_tf_type = seq_len_tf.soft_get('data_type', None)
        seq_len_tf_name = seq_len_tf.soft_get('name', seq_len_tf.id)
        seq_mask_placeholder = Parameter(graph, {'name': seq_len_tf_name, 'shape': seq_len_tf_shape,
                                                 'data_type': seq_len_tf_type}).create_node()
        reduce_to_seq_len_node = create_op_with_const_inputs(graph, ReduceSum, {1: np.array(1, dtype=np.int32)},
                                                             {'name': seq_len_tf_name + '/ReduceToSeqLen',
                                                              'keep_dims': False})
        reduce_to_seq_len_node.in_port(0).connect(seq_mask_placeholder.out_port(0))
        seq_len_tf.out_port(0).get_connection().set_source(reduce_to_seq_len_node.out_port(0))

        cast_fp_type = data_type_str_to_np(graph.graph['cmd_params'].data_type)
        casted_seq_mask_node = Cast(graph, {'name': seq_len_tf_name + '/CastToFP32', 'dst_type': cast_fp_type}).create_node()
        casted_seq_mask_node.in_port(0).connect(seq_mask_placeholder.out_port(0))
        permuted_casted_seq_mask = create_op_with_const_inputs(graph, Transpose, {1: int64_array([1, 0])},
                                                               {'name': seq_len_tf_name + '/Permute'})
        permuted_casted_seq_mask.in_port(0).connect(casted_seq_mask_node.out_port(0))
        rename_nodes([(seq_len_tf, seq_len_tf_name + '/AbandonedName'), (seq_mask_placeholder, seq_len_tf_name)])

        # create CTCGreedyDecoder node and set mask node
        ctc_merge_repeated_i = ctc_greedy_decoder_tf.soft_get('ctc_merge_repeated', ctc_greedy_decoder_tf.id)
        ctc_greedy_decoder = CTCGreedyDecoderOp(graph, {'name': output_sparse_to_dense_name,
                                                        'ctc_merge_repeated': ctc_merge_repeated_i}).create_node()
        ctc_greedy_decoder.in_port(1).connect(permuted_casted_seq_mask.out_port(0))
        rename_nodes([(sparse_to_dense_tf, output_sparse_to_dense_name + '/AbandonedName'),
                      (ctc_greedy_decoder, output_sparse_to_dense_name)])

        # create CTCLoss node and set attributes
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

        # connect logits
        ctc_greedy_decoder_tf.in_port(0).get_connection().set_destination(ctc_greedy_decoder.in_port(0))
        ctc_loss.in_port(0).disconnect()
        transpose_tf.in_port(0).get_connection().add_destination(ctc_loss.in_port(0))

        # connect logit lengths
        ctc_greedy_decoder_tf.in_port(1).disconnect()
        ctc_loss.in_port(1).connect(reduce_to_seq_len_node.out_port(0))

        # connect labels to ctc_loss
        squeeze_op = create_op_with_const_inputs(graph, Squeeze, {1: int64_array([2, 3])})
        cast_labels_op = Cast(graph, {'name': output_sparse_to_dense_name + '/CastLabels', 'dst_type': np.int32}).create_node()
        squeeze_op.in_port(0).connect(ctc_greedy_decoder.out_port(0))
        cast_labels_op.in_port(0).connect(squeeze_op.out_port(0))
        ctc_loss.in_port(2).connect(cast_labels_op.out_port(0))

        # connect label lengths
        equal_op = create_op_with_const_inputs(graph, Equal, {1: np.array([-1], dtype=np.int32)},
                                               {'name': output_sparse_to_dense_name + '/Equal'})
        equal_op.in_port(0).connect(cast_labels_op.out_port(0))
        labels_shape_op = Shape(graph, {'name': output_sparse_to_dense_name + '/ShapeOf'}).create_node()
        labels_shape_op.in_port(0).connect(equal_op.out_port(0))
        broadcast_one = create_op_with_const_inputs(graph, Broadcast, {0: np.array([1], dtype=np.int32)},
                                                    {'mode': 'numpy',
                                                     'name': output_sparse_to_dense_name + '/One'})
        broadcast_one.in_port(1).connect(labels_shape_op.out_port(0))
        broadcast_zero = create_op_with_const_inputs(graph, Broadcast, {0: np.array([0], dtype=np.int32)},
                                                     {'mode': 'numpy',
                                                      'name': output_sparse_to_dense_name + '/Zero'})
        broadcast_zero.in_port(1).connect(labels_shape_op.out_port(0))

        select_node = Select(graph, {'name': output_sparse_to_dense_name + '/Select'}).create_node()
        select_node.in_port(0).connect(equal_op.out_port(0))
        select_node.in_port(1).connect(broadcast_zero.out_port(0))
        select_node.in_port(2).connect(broadcast_one.out_port(0))
        label_length_node = create_op_with_const_inputs(graph, ReduceSum, {1: int64_array([1])},
                                                      op_attrs={'name': output_sparse_to_dense_name + '/LabelLength',
                                                                'keep_dims': False})
        label_length_node.in_port(0).connect(select_node.out_port(0))
        ctc_loss.in_port(3).connect(label_length_node.out_port(0))

        # set source for output of new sub-graph and remove old nodes
        ctc_loss_tf.out_port(0).get_connection().set_source(ctc_loss.out_port(0))
        graph.remove_nodes_from([ctc_greedy_decoder_tf.id, ctc_loss_tf.id, cast_tf.id, sparse_to_dense_tf.id])
