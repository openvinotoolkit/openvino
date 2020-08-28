"""
 Copyright (C) 2018-2020 Intel Corporation

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

import logging as log

import numpy as np

from extensions.ops.Cast import Cast
from extensions.front.Pack import Pack
from extensions.front.FillToBroadcast import FillToBroadcast
from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, rename_nodes
from mo.ops.broadcast import Broadcast
from mo.ops.concat import Concat
from mo.ops.squeeze import Squeeze
from mo.ops.unsqueeze import Unsqueeze


class CTCGreedyDecoderReplacement(FrontReplacementSubgraph):
    """
    TensorFlow CTCGreedyDecoder produces output in a sparse tensor that is not supported by Inference Engine and
    Inference Engine's CTCGreedyDecoder has different output that is in a dense format. So this transformation
    intents to replace TF CTCGreedyDecoder+SparseToDense with IE one.
    Also Inference Engine's CTCGreedyDecoder has a specific format for the second input tensor, a sequence length,
    different from TF's one so this transformation cares about transformation of its format.
    The second input to the CTCGreedyDecoder in the TensorFlow is a 1D tensor with sequence lengths. In the Inference
    Engine the second input to the CTCGreedyDecoder is a 2D tensor, a sequence mask, where the first element
    in each row is equal to 1 and all others in the tail are equal to 0. The number of ones represents
    a sequence length.
    """
    enabled = True

    def run_after(self):
        # CTCGreedyDecoderReplacement is not reshape-able transformation
        # so reshape-able CTCGreedyDecoderReplacement2 transformation is applied first
        return [CTCGreedyDecoderReplacement2]

    @staticmethod
    def pattern(**kwargs):
        return dict(
            nodes=[('decoder', dict(op='CTCGreedyDecoder')),
                   ('cast', dict(op='Cast')),
                   ('sparse_to_dense', dict(op='SparseToDense'))
                   ],
            edges=[('decoder', 'sparse_to_dense', {'out': 0}),
                   ('decoder', 'cast', {'out': 1}),
                   ('cast', 'sparse_to_dense', {'out': 0})
                   ]
        )

    def replace_sub_graph(self, graph: Graph, match: dict):
        # TODO: Once Inference Engine's CTCGreedyDecoder starts to support sequence length format like in TensorFlow,
        # CTCGreedyDecoderReplacement2 needs to be removed and CTCGreedyDecoderReplacement, a more generic
        # transformation, needs to be adopted for all cases
        ctc_greedy_decoder = match['decoder']
        cast = match['cast']
        sparse_to_dense = match['sparse_to_dense']
        sparse_to_dense_name = sparse_to_dense.soft_get('name', sparse_to_dense.id)

        # disconnect SparseToDense and Cast nodes
        sparse_to_dense.in_port(0).disconnect()
        cast.in_port(0).disconnect()

        # transform CTCGreedyDecoder output to TensorFlow's one:
        # 1. squeeze the output to [N, T] shape
        # 2. cast it to integer
        squeeze_dec_seq = create_op_with_const_inputs(graph, Squeeze, {1: int64_array([2, 3])},
                                                      {'name': sparse_to_dense_name})
        squeeze_dec_seq.in_port(0).connect(ctc_greedy_decoder.out_port(0))
        cast_to_int = Cast(graph, {'name': sparse_to_dense_name + '/CastToInt',
                                   'dst_type': np.int32}).create_node()
        cast_to_int.in_port(0).connect(squeeze_dec_seq.out_port(0))

        # preserve output name from original graph
        rename_nodes([(sparse_to_dense, sparse_to_dense_name + '/AbandonedName'),
                      (cast_to_int, sparse_to_dense_name)])

        # set output of the new sub-graph as a source for SparseToDense consumer
        sparse_to_dense.out_port(0).get_connection().set_source(cast_to_int.out_port(0))

        # remove no longer needed nodes
        graph.remove_nodes_from([sparse_to_dense.id, cast.id])

        # mark CTCGreedyDecoder node as a node that requires transformation of sequence length to a mask format
        # in the middle phase
        ctc_greedy_decoder['use_mask_format'] = True

        # unless the second input of CTCGreedyDecoder is a parameter, it enforces MO to use --static-shape
        # to try getting the second input with a value
        sequence_length_node = ctc_greedy_decoder.in_node(1)
        if sequence_length_node.soft_get('op') != 'Parameter' and not graph.graph['cmd_params'].static_shape:
            log.error(
                "Model can not be translated in a reshape-able way.\n"
                "Model Optimizer key static_shape was turned on to prevent related errors.\n"
                "There will be no success changing input shapes of the model with the help of "
                "InferenceEngine reshape method", extra={'is_warning': True})
            graph.graph['cmd_params'].static_shape = True


class CTCGreedyDecoderReplacement2(FrontReplacementSubgraph):
    """
    The TF implementation of the CTCGreedyDecoder produces a tuple with two tensors. The first element in the tuple is
    the SparseTensor which is converted to a regular tensor with the SparseToDense operation. This replacer matches
    CTCGreedyDecoder and SparseToDense operations and removes the SparseToDense and Cast operation which is also used
    in the SparseToDense operation, because Inference Engine implementation of the CTCGreedyDecoder produces regular
    tensor as output.
    Also, Inference Engine CTCGreedyDecoder requires a mask format for sequence lengths that is a different from
    original one. Hence, this transformation changes a format of sequence length to a mask by replacing Fill and Pack
    nodes with a special graph that produces a tensor of ones with shape [T, N] accepted by opset CTCGreedyDecoder.
    """
    enabled = True

    def run_before(self):
        return [Pack, FillToBroadcast]

    @staticmethod
    def pattern(**kwargs):
        return dict(
            nodes=[
                ('transpose', dict(op='Transpose')),
                ('shape', dict(op='ShapeOf')),
                ('shape_1', dict(op='ShapeOf')),
                ('strided_slice', dict(op='StridedSlice')),
                ('stack', dict(op='Const', value=lambda v: v is not None and np.array_equal(v, [1]))),
                ('stack1', dict(op='Const', value=lambda v: v is not None and np.array_equal(v, [2]))),
                ('stack2', dict(op='Const', value=lambda v: v is not None and np.array_equal(v, [1]))),
                ('strided_slice_1', dict(op='StridedSlice')),
                ('stack_1', dict(op='Const', value=lambda v: v is not None and np.array_equal(v, [0]))),
                ('stack1_1', dict(op='Const', value=lambda v: v is not None and np.array_equal(v, [1]))),
                ('stack2_1', dict(op='Const', value=lambda v: v is not None and np.array_equal(v, [1]))),
                ('dims', dict(op='Pack')),
                ('fill', dict(op='Fill')),
                ('decoder', dict(op='CTCGreedyDecoder')),
                ('cast', dict(op='Cast')),
                ('sparse_to_dense', dict(op='SparseToDense')),
            ],
            edges=[
                ('transpose', 'shape', {'out': 0}),
                ('transpose', 'shape_1', {'out': 0}),
                ('transpose', 'decoder', {'out': 0, 'in': 0}),
                ('shape', 'strided_slice', {'out': 0, 'in': 0}),
                ('stack', 'strided_slice', {'out': 0, 'in': 1}),
                ('stack1', 'strided_slice', {'out': 0, 'in': 2}),
                ('stack2', 'strided_slice', {'out': 0, 'in': 3}),
                ('shape_1', 'strided_slice_1', {'out': 0, 'in': 0}),
                ('stack_1', 'strided_slice_1', {'out': 0, 'in': 1}),
                ('stack1_1', 'strided_slice_1', {'out': 0, 'in': 2}),
                ('stack2_1', 'strided_slice_1', {'out': 0, 'in': 3}),
                ('strided_slice', 'dims', {'out': 0, 'in': 0}),
                ('dims', 'fill', {'out': 0, 'in': 0}),
                ('strided_slice_1', 'fill', {'out': 0, 'in': 1}),
                ('fill', 'decoder', {'out': 0, 'in': 1}),
                ('decoder', 'sparse_to_dense', {'out': 0, 'in': 0}),
                ('decoder', 'cast', {'out': 1, 'in': 0}),
                ('cast', 'sparse_to_dense', {'out': 0}),
            ]
        )

    def replace_sub_graph(self, graph: Graph, match: dict):
        # obtain references to necessary nodes and their names
        fill = match['fill']
        dims = match['dims']
        strided_slice = match['strided_slice']
        strided_slice_1 = match['strided_slice_1']
        ctc_greedy_decoder = match['decoder']
        cast = match['cast']
        sparse_to_dense = match['sparse_to_dense']
        strided_slice_name = strided_slice.soft_get('name', strided_slice.id)
        strided_slice_1_name = strided_slice_1.soft_get('name', strided_slice_1.id)
        ctc_greedy_decoder_name = ctc_greedy_decoder.soft_get('name', ctc_greedy_decoder.id)
        sparse_to_dense_name = sparse_to_dense.soft_get('name', sparse_to_dense.id)

        # unsqueeze scalar values with batch size and time dimension
        unsqueeze_batch_size = create_op_with_const_inputs(graph, Unsqueeze, {1: int64_array(0)},
                                                           {'name': strided_slice_name + '/Unsqueeze'})
        dims.in_port(0).get_connection().set_destination(unsqueeze_batch_size.in_port(0))
        unsqueeze_time_size = create_op_with_const_inputs(graph, Unsqueeze, {1: int64_array(0)},
                                                           {'name': strided_slice_1_name + '/Unsqueeze'})
        fill.in_port(1).get_connection().set_destination(unsqueeze_time_size.in_port(0))

        # compute a sequence mask shape [T, N] required for CTCGreedyDecoder
        seq_mask_shape = Concat(graph, {'axis': 0, 'in_ports_count': 2,
                                        'name': ctc_greedy_decoder_name + '/SequenceMaskShape'}).create_node()
        seq_mask_shape.in_port(0).connect(unsqueeze_time_size.out_port(0))
        seq_mask_shape.in_port(1).connect(unsqueeze_batch_size.out_port(0))

        # compute a sequence mask
        sequence_mask = create_op_with_const_inputs(graph, Broadcast, {0: np.array([1.0], dtype=np.float)},
                                                    {'mode': 'numpy',
                                                     'name': ctc_greedy_decoder_name + '/SequenceMask'})
        sequence_mask.in_port(1).connect(seq_mask_shape.out_port(0))

        # create CTCGreedyDecoder with the sequence mask instead of sequence length
        ctc_greedy_decoder.in_port(1).disconnect()
        ctc_greedy_decoder.in_port(1).connect(sequence_mask.out_port(0))

        # remove fill and pack nodes since they are now in unconnected component
        graph.remove_nodes_from([fill.id, dims.id])

        # transform opset CTCGreedyDecoder output to TensorFlow's one that has a shape [N, T]
        # opset CTCGreedyDecoder has an output with a shape [N, T, 1, 1]
        squeeze_dec_seq = create_op_with_const_inputs(graph, Squeeze, {1: int64_array([2, 3])},
                                                      {'name': sparse_to_dense_name})
        squeeze_dec_seq.in_port(0).connect(ctc_greedy_decoder.out_port(0))
        cast_to_int = Cast(graph, {'name': sparse_to_dense_name + '/CastToInt',
                                   'dst_type': np.int32}).create_node()
        cast_to_int.in_port(0).connect(squeeze_dec_seq.out_port(0))

        # preserve output name from original graph
        rename_nodes([(sparse_to_dense, sparse_to_dense_name + '/AbandonedName'),
                      (cast_to_int, sparse_to_dense_name)])

        # set output of the new sub-graph as a source for SparseToDense consumer
        sparse_to_dense.out_port(0).get_connection().set_source(cast_to_int.out_port(0))

        # cleanup a graph
        graph.remove_nodes_from([cast.id, sparse_to_dense.id])
