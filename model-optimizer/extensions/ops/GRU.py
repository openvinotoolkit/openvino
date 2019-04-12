"""
 Copyright (c) 2019 Intel Corporation

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
from extensions.ops.RNN import rnn_infer
from mo.graph.graph import Node, Graph
from mo.ops.op import Op
import numpy as np


class GRU(Op):
    op = 'GRU'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': 'RNNSequence',  # should be never emitted to IR; for debugging purposes
            'op': __class__.op,
            'blobs_wrb': False,
            'has_num_directions': False,
            'direction': 'forward',
            'infer': __class__.infer,
            'multiplier': 3,
            'multilayers': False,
            'gate_order': np.array([0, 1, 2]),  # TODO: change it later
            'normalized': False,

            'activation_alpha': None,
            'activation_beta': None,
            'activations': None,
            'clip': None,
            'linear_before_reset': None,
            'in_ports_count': 6,
            'out_ports_count': 2,
        }
        super().__init__(graph, mandatory_props, attrs)

    @staticmethod
    def supported_attrs():
        return [
            'hidden_size',  # number of the elements in hidden cell size
            'direction',  # one of 'forward', 'reverse', or 'bidirectional'
            'axis',

            'activation_alpha',
            'activation_beta',
            'activations',
            'clip',
            'linear_before_reset',
        ]

    def backend_attrs(self):
        return [
            'hidden_size',  # number of the elements in hidden cell size
            'direction',  # one of 'forward', 'reverse', or 'bidirectional'
            'axis',

            'activation_alpha',
            'activation_beta',
            ('activations', lambda node: ','.join(node.activations) if node.activations is not None else None),
            'clip',
            'linear_before_reset',
        ]

    @staticmethod
    def infer(node: Node):
        assert len(node.in_nodes()) >= 3  # X, W and R
        assert len(node.in_nodes()) <= 5
        assert len(node.out_nodes()) <= 2

        rnn_infer(node, [1])
