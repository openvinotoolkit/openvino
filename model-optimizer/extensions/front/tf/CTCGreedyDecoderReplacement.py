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

from mo.front.common.replacement import FrontReplacementSubgraph
from mo.graph.graph import Graph
from mo.ops.const import Const
from mo.utils.error import Error


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
        # TODO: it requires further refactoring and improvement to provide reshape-ability
        decoder_node = match['decoder']
        decoder_node_name = decoder_node.soft_get('name', decoder_node.id)
        graph.remove_edge(decoder_node.id, match['sparse_to_dense'].id)
        graph.remove_edge(decoder_node.id, match['cast'].id)
        match['sparse_to_dense'].replace_node(decoder_node)

        sequence_length_node = decoder_node.in_node(1)
        if sequence_length_node.value is None:
            raise Error('The second input to the CTCGreedyDecoder node "{}" is not constant. This case is not '
                        'supported with the Inference Engine.'.format(decoder_node_name))

        # the batch size is the dimension with index 1 for the layer CTCGreedyDecoder
        mask_value = np.ones([decoder_node.in_node(0).shape[1], sequence_length_node.value[0]])
        mask_value[:, 0] = 0
        mask_value = np.transpose(mask_value)
        mask_node = Const(graph, {'name': decoder_node_name + '/Mask',
                                  'value': mask_value}).create_node()
        decoder_node.in_port(1).disconnect()
        decoder_node.in_port(1).connect(mask_node.out_port(0))

        return {}
