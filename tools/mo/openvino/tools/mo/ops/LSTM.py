# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.RNN import rnn_infer, RNN
from openvino.tools.mo.ops.op import Op


class LSTM(Op):
    op = 'LSTM'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': 'RNNSequence',  # should be never emitted to IR; for debugging purposes
            'op': self.op,
            'blobs_wrb': False,  # input blobs have three separate components W, R and B like in ONNX/LSTM
            'has_num_directions': False,  # if True, output shape has 4 dimensions; 3D otherwise
            'direction': 'forward',
            'infer': self.infer,
            'reverse_infer': RNN.reverse_infer,
            'multiplier': 4,
            'gate_order': None,
            'normalized': False,
            'multilayers': False,
            'format': None,  # format type of input blobs for different frameworks (onnx, tf),

            'activation_alpha': None,
            'activation_beta': None,
            'activations': None,
            'clip': None,
            'input_forget': None,
            'in_ports_count': 7,
            'out_ports_count': 3,
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
            # 'input_forget',  # Not supported yet
        ]

    def backend_attrs(self):
        return [
            'hidden_size',  # number of the elements in hidden cell size
            'direction',  # one of 'forward', 'reverse', or 'bidirectional'
            'axis',

            ('activations', lambda node: ','.join(node['activations']) if node.has_and_set('activations') else None),
            ('activations_alpha', lambda node: ','.join(map(str, node['activations_alpha']))
                if node.has_and_set('activations_alpha') else None),
            ('activations_beta', lambda node: ','.join(map(str, node['activations_beta']))
                if node.has_and_set('activations_beta') else None),
            'clip',
            # 'input_forget',  # Not supported yet
        ]

    @staticmethod
    def infer(node: Node):
        # there are limitations coming from ONNX LSTM definition and normalization rules
        assert len(node.in_nodes()) >= 3  # X, W and R
        assert len(node.in_nodes()) <= 7
        assert len(node.out_nodes()) <= 3

        rnn_infer(node, [1, 2])
