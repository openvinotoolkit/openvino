"""
 Copyright (c) 2017-2018 Intel Corporation

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

import networkx as nx
import numpy as np

from mo.front.common.partial_infer.utils import mark_input_bins
from mo.graph.graph import Node
from mo.ops.op import Op
from mo.utils.utils import refer_to_faq_msg


class LSTMSequence(Op):
    ''' Implements a layer that incorporates LSTM cell in a loop like it is specified in ONNX

        It is assumed that there is no equivalent of this op in IE,
        so it is considered as intermediate operation that will be translated differently.
        We define type for this operation to enable debuggin at IE side.

        There are several flavors of this op depending on how it was created:
            - ONNX/LSTM gives this op in non-normalized form and will require normalization
                as a separate transformation (see LSTMSequenceNormalize middle transformation);
                in this case blobs_wrb=True
            - other sources should give already normalized operation (with blobs_wrb=False).
    '''
    op = 'LSTMSequence'

    def __init__(self, graph: nx.MultiDiGraph, attrs: dict):
        mandatory_props = {
            'type': '__LSTMSequence',
            'op': __class__.op,
            'blobs_wrb': False,
            'infer': __class__.infer
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return [
            'hidden_size',  # number of the elements in hidden cell size
            'batch_dim',    # batch dimension index in input/output shape
            'sequence_dim', # sequence dimension index in input/output shape
            'blobs_wrb',    # input blobs have three separate components W, R and B like in ONNX/LSTM
        ]

    def backend_attrs(self):
        return [
            'hidden_size',
        ]

    @staticmethod
    def infer(node: Node):
        # there are limitations coming from ONNX LSTM definition and normalization rules
        assert len(node.in_nodes()) >= 3
        assert len(node.in_nodes()) <= 7
        assert len(node.out_nodes()) <= 3
        assert node.batch_dim <= 1
        assert node.sequence_dim <=1
        assert node.batch_dim != node.sequence_dim

        if node.blobs_wrb:
            mark_input_bins(node, ['W', 'R', 'B'])
        else:
            mark_input_bins(node)
        input_shape = node.in_node(0).shape
        assert len(input_shape) == 3
        node.out_node(0).shape = np.array([input_shape[0], input_shape[1], node.hidden_size], dtype=np.int64)
        if len(node.out_nodes()) > 1:
            state_size = np.array([input_shape[1], node.hidden_size], dtype=np.int64)
            node.out_node(1).shape = state_size.copy()
            if len(node.out_nodes()) > 2:
                node.out_node(2).shape = state_size.copy()
