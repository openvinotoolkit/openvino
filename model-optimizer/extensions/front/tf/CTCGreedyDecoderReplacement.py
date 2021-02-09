"""
 Copyright (C) 2018-2021 Intel Corporation

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

from extensions.front.FillToBroadcast import FillToBroadcast
from extensions.front.Pack import Pack
from extensions.ops.Cast import Cast
from extensions.ops.ctc_greedy_decoder_seq_len import CTCGreedyDecoderSeqLenOp
from extensions.ops.transpose import Transpose
from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, rename_nodes
from mo.ops.broadcast import Broadcast
from mo.ops.concat import Concat
from mo.ops.result import Result
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

    @staticmethod
    def pattern(**kwargs):
        return dict(
            nodes=[('decoder', dict(op='CTCGreedyDecoderSeqLen')),
                   ('cast', dict(op='Cast')),
                   ('sparse_to_dense', dict(op='SparseToDense'))
                   ],
            edges=[('decoder', 'sparse_to_dense', {'out': 0}),
                   ('decoder', 'sparse_to_dense', {'out': 2}),
                   ('decoder', 'cast', {'out': 1}),
                   ('cast', 'sparse_to_dense', {'out': 0})
                   ]
        )

    def replace_sub_graph(self, graph: Graph, match: dict):
        # TODO: Once Inference Engine's CTCGreedyDecoder starts to support sequence length format like in TensorFlow,
        ctc_greedy_decoder_tf = match['decoder']
        cast = match['cast']
        sparse_to_dense = match['sparse_to_dense']

        ctc_data_permute = create_op_with_const_inputs(graph, Transpose, {1: int64_array([1, 0, 2])},
                                                            {'name': ctc_greedy_decoder_tf.name + '/ctc_data_permute'})
        ctc_greedy_decoder_tf.in_port(0).get_source().connect(ctc_data_permute.in_port(0))
        merge_repeated_tf = ctc_greedy_decoder_tf.soft_get('merge_repeated', ctc_greedy_decoder_tf.id)
        ctc_greedy_decoder = CTCGreedyDecoderSeqLenOp(graph, {'name': ctc_greedy_decoder_tf.name,
                                                              'cmerge_repeated': merge_repeated_tf}).create_node()
        ctc_greedy_decoder.in_port(0).connect(ctc_data_permute.out_port(0))
        ctc_greedy_decoder_tf.in_port(1).get_source().connect(ctc_greedy_decoder.in_port(1))
        ctc_res1 = Result(graph, {'name': ctc_greedy_decoder_tf.soft_get('name', ctc_greedy_decoder_tf.id) + '/Result'}).create_node()
        # set output of the new sub-graph as a source for SparseToDense consumer
        sparse_to_dense.out_port(0).get_connection().set_source(ctc_greedy_decoder.out_port(0))
        ctc_res1.in_port(0).get_connection().set_source(ctc_greedy_decoder.out_port(1))

        # remove no longer needed nodes
        graph.remove_nodes_from([sparse_to_dense.id, cast.id, ctc_greedy_decoder_tf.id])

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
