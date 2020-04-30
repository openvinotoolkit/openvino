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
from extensions.ops.ReduceOps import ReduceSum
from extensions.ops.select import Select
from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, rename_nodes
from mo.ops.broadcast import Broadcast
from mo.ops.const import Const
from mo.ops.shape import Shape
from mo.ops.squeeze import Squeeze
from mo.utils.error import Error


class CTCLossFrontReplacement(FrontReplacementSubgraph):
    """
    The CTCLoss appears along with CTCGreedyDecoder operation in particular. Since the TF CTCGreedyDecoder outputs
    sparse tensor format, the OpenVINO CTCGreedyDecoder has a different format and the CTCLoss is also affected
    in terms of different formar for its inputs. So the corresponding sub-graph with CTCGreedyDecoding and CTCLoss
    must be transformed properly.
    """
    enabled = True

    def run_before(self):
        from extensions.front.tf.CTCGreedyDecoderReplacement import CTCGreedyDecoderReplacement
        return [CTCGreedyDecoderReplacement]

    def pattern(self):
        return dict(
            nodes=[
                ('transpose', dict(op='Transpose')),
                ('ctc_greedy_decoder', dict(op='CTCGreedyDecoder')),
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
        output_ctc_loss_name = ctc_loss_tf.soft_get('name', ctc_loss_tf.id)
        ctc_greedy_decoder_tf_name = ctc_greedy_decoder_tf.soft_get('name', ctc_greedy_decoder_tf.id)

        log.debug('Found CTCLossFrontReplacer pattern after {} with name {}'.format(ctc_greedy_decoder_tf.op,
                                                                                    ctc_greedy_decoder_tf.name))

        # create CTCGreedyDecoder node and set mask node
        ctc_merge_repeated_i = ctc_greedy_decoder_tf.soft_get('ctc_merge_repeated', ctc_greedy_decoder_tf.id)
        ctc_greedy_decoder = CTCGreedyDecoderOp(graph, {'name': output_sparse_to_dense_name,
                                                        'ctc_merge_repeated': ctc_merge_repeated_i}).create_node()
        sequence_length_node = ctc_greedy_decoder_tf.in_node(1)
        if sequence_length_node.value is None:
            raise Error('The second input to the CTCGreedyDecoder node "{}" is not constant. This case is not '
                        'supported with the Inference Engine.'.format(ctc_greedy_decoder_tf_name))
        batch_size = sequence_length_node.value.shape[0]
        mask_value = np.ones([batch_size, sequence_length_node.value[0]])
        mask_value = np.transpose(mask_value)
        mask_node = Const(graph, {'name': output_ctc_loss_name + '/Mask',
                                  'value': mask_value}).create_node()
        ctc_greedy_decoder.in_port(1).connect(mask_node.out_port(0))
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
        transpose_tf.in_port(0).get_connection().add_destination(ctc_loss.in_port(0))

        # connect logit lengths
        ctc_greedy_decoder_tf.in_port(1).disconnect()
        ctc_loss_tf.in_port(3).get_connection().set_destination(ctc_loss.in_port(1))

        # connect labels to ctc_loss
        squeeze_op = create_op_with_const_inputs(graph, Squeeze, {1: int64_array([2, 3])})
        cast_labels_op = Cast(graph, {'name': output_sparse_to_dense_name + '/CastLabels', 'dst_type': np.int32}).create_node()
        squeeze_op.in_port(0).connect(ctc_greedy_decoder.out_port(0))
        cast_labels_op.in_port(0).connect(squeeze_op.out_port(0))
        ctc_loss.in_port(2).connect(cast_labels_op.out_port(0))

        # connect label lengths
        equal_op = Equal(graph, {'name': output_sparse_to_dense_name + '/Equal'}).create_node()
        minus_one_node = Const(graph, {'name': output_sparse_to_dense_name + '/MinusOne',
                                       'value': np.array([-1], dtype=np.int32)}).create_node()
        equal_op.in_port(0).connect(cast_labels_op.out_port(0))
        equal_op.in_port(1).connect(minus_one_node.out_port(0))
        labels_shape_op = Shape(graph, {'name': output_sparse_to_dense_name + '/ShapeOf'}).create_node()
        labels_shape_op.in_port(0).connect(equal_op.out_port(0))
        broadcast_one = create_op_with_const_inputs(graph, Broadcast, {0: np.array([1], dtype=np.int32)},
                                                    {'name': output_sparse_to_dense_name + '/One'})
        broadcast_one.in_port(1).connect(labels_shape_op.out_port(0))
        broadcast_zero = create_op_with_const_inputs(graph, Broadcast, {0: np.array([0], dtype=np.int32)},
                                                    {'name': output_sparse_to_dense_name + '/Zero'})
        broadcast_zero.in_port(1).connect(labels_shape_op.out_port(0))

        select_node = Select(graph, {'name': output_sparse_to_dense_name + '/Select'}).create_node()
        select_node.in_port(0).connect(equal_op.out_port(0))
        select_node.in_port(1).connect(broadcast_one.out_port(0))
        select_node.in_port(2).connect(broadcast_zero.out_port(0))
        label_length_node = create_op_with_const_inputs(graph, ReduceSum, {1: int64_array([1])},
                                                      op_attrs={'name': output_sparse_to_dense_name + '/LabelLength',
                                                                'keep_dims': False})
        label_length_node.in_port(0).connect(select_node.out_port(0))
        ctc_loss.in_port(3).connect(label_length_node.out_port(0))

        # set source for output of new sub-graph and remove old nodes
        ctc_loss_tf.out_port(0).get_connection().set_source(ctc_loss.out_port(0))
        graph.remove_nodes_from([ctc_greedy_decoder_tf.id, ctc_loss_tf.id, cast_tf.id, sparse_to_dense_tf.id])
