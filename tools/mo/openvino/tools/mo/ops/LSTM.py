# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import shape_array, dynamic_dimension_value
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.RNN import rnn_infer
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
            'reverse_infer': self.reverse_infer,
            'multiplier': 4,
            'gate_order': None,
            'normalized': False,
            'multilayers': False,
            'format': None,  # format type of input blobs for different frameworks (onnx, tf, mxnet),

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

            'activation_alpha',
            'activation_beta',
            ('activations', lambda node: ','.join(node.activations) if node.activations is not None else None),
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

    @staticmethod
    def reverse_infer(node: Node):
        W_size = np.prod(node.in_port(1).data.get_shape())

        multiplier = node.multiplier
        hidden_size = node.hidden_size
        num_layers = node.num_layers
        direction = 2 if node.has_num_directions else 1

        size = hidden_size * direction * multiplier
        other_layer_params_size = (hidden_size * direction + hidden_size + 2) * size
        first_layer_params_size = W_size - (num_layers - 1) * other_layer_params_size

        batch_size = 1
        seq_len = 1
        # input_size can be determined from the first_layer_params_size (e.g. MXNetSplitMultiLayers.py:79)
        # if first_layer_params_size = (input_size + hidden_size + 2) * size
        # then input_size = first_layer_params_size / size - 2 - hidden_size
        input_size = first_layer_params_size / size - 2 - hidden_size
        if node.is_in_port_connected(3):
            initial_cell_state_size = node.in_port(3).data.get_shape()
            if initial_cell_state_size is not None:
                batch_size = initial_cell_state_size[1]

        input_shape = shape_array([seq_len, batch_size, input_size])
        node.in_port(0).data.set_shape(input_shape)
