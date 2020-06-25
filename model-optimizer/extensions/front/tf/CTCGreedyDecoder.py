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

import numpy as np

from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.graph.graph import Node, Graph
from mo.utils.error import Error

from mo.ops.broadcast import Broadcast
from mo.ops.shape import Shape
from mo.ops.squeeze import Squeeze
from mo.ops.unsqueeze import Unsqueeze
from extensions.ops.gather import Gather
from mo.ops.concat import Concat
from mo.ops.strided_slice import StridedSlice
from mo.front.tf.graph_utils import create_op_node_with_second_input, create_op_with_const_inputs


class CTCGreedyDecoderReplacement(FrontReplacementSubgraph):
    """
    The TF implementation of the CTCGreedyDecoder produces a tuple with two tensors. The first element in the tuple is
    the SparseTensor which is converted to a regular tensor with the SparseToDense operation. This replacer matches
    CTCGreedyDecoder and SparseToDense operations and removes the SparseToDense and Cast operation which is also used
    in the SparseToDense operation, because Inference Engine implementation of the CTCGreedyDecoder produces regular
    tensor as output.

    The second input to the CTCGreedyDecoder in the TensorFlow is a 1D tensor with sequence lengths. In the Inference
    Engine the second input to the CTCGreedyDecoder is a 2D tensor where the first element in each row is equal to 0
    and all others are equal to 1. The length of the row is equal to the sequence length. The replacer modifies the
    second input to be compatible with the Inference Engine CTCGreedyDecoder layer implementation.
    """
    enabled = True

    @staticmethod
    def pattern(**kwargs):
        return dict(
            nodes=[
                ('decoder', dict(op='CTCGreedyDecoder')),
                ('cast', dict(op='Cast')),
                ('sparse_to_dense', dict(op='SparseToDense')),
            ],
            edges=[
                ('decoder', 'sparse_to_dense', {'out': 0}),
                ('decoder', 'cast', {'out': 1}),
                ('cast', 'sparse_to_dense', {'out': 0}),
            ]
        )

    def nodes_to_remove(self, graph: Graph, match: dict):
        return [match['cast'].id, match['sparse_to_dense']]

    def replace_sub_graph(self, graph: Graph, match: dict):
        decoder_node = match['decoder']
        graph.remove_edge(decoder_node.id, match['sparse_to_dense'].id)
        graph.remove_edge(decoder_node.id, match['cast'].id)
        match['sparse_to_dense'].replace_node(decoder_node)

        decoder_name = decoder_node.soft_get('name', decoder_node.id)

        shape = Shape(graph, {'name': decoder_name + '/shape'}).create_node()

        ss = create_op_with_const_inputs(graph, StridedSlice, {1: int64_array([1]),
                                                               2: int64_array([2]),
                                                               3: int64_array([1])},
                                                               {'name': decoder_name + '/strided_slice',
                                                                'begin_mask': '1',
                                                                'ellipsis_mask': '0',
                                                                'end_mask': '1',
                                                                'new_axis_mask': '0',
                                                                'shrink_axis_mask': '1'})

        decoder_node.in_port(0).get_source().connect(shape.in_port(0))
        shape.out_port(0).connect(ss.in_port(0))

        unsqueeze_1 = create_op_node_with_second_input(graph, Unsqueeze, int64_array([0, 1]), {'name': decoder_name + '/unsqueeze_1'})
        unsqueeze_1.in_port(0).connect(ss.out_port(0))

        unsqueeze_2 = create_op_node_with_second_input(graph, Unsqueeze, int64_array([0]), {'name': decoder_name + '/unsqueeze_2'})

        port = decoder_node.in_port(1).get_source()
        port.disconnect()
        port.connect(unsqueeze_2.in_port(0))

        concat = Concat(graph, {'name': decoder_name + '/concat'}).create_node()
        concat.add_input_port(0, skip_if_exist=True)
        concat.add_input_port(1, skip_if_exist=True)

        concat.in_port(1).connect(unsqueeze_1.out_port(0))
        concat.in_port(0).connect(unsqueeze_2.out_port(0))

        broadcast = create_op_with_const_inputs(graph, Broadcast, {0: int64_array([1])}, {'name': decoder_name + '/broadcast'})

        squeeze = create_op_node_with_second_input(graph, Squeeze, int64_array([0]), {'name': decoder_name + '/squeeze'})
        concat.out_port(0).connect(squeeze.in_port(0))
        squeeze.out_port(0).connect(broadcast.in_port(1))

        broadcast.out_port(0).connect(decoder_node.in_port(1))

        return {}
