# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.GRU import GRU
from openvino.tools.mo.ops.LSTM import LSTM
from openvino.tools.mo.ops.RNN import RNN
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs
from openvino.tools.mo.utils.error import Error
from openvino.tools.mo.utils.utils import refer_to_faq_msg


class RNNFrontExtractor(FrontExtractorOp):
    op = 'RNN'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        mode = attrs.str('mode', None)
        state_size = attrs.int('state_size', None)
        bidirectional = attrs.bool('bidirectional', False)
        num_layers = attrs.int('num_layers', 1)
        layout = attrs.str('layout', 'TNC')  # in MXNet RNN by default take data in
        # format [seq_len, batch_size, inp_size]

        node_attrs = {
            'batch_dim': layout.index('N'),
            'sequence_dim': layout.index('T'),
            'blobs_wrb': False,
            'hidden_size': state_size,
            'has_num_directions': bidirectional,
            'direction': 'bidirectional' if bidirectional else 'forward',
            'num_layers': num_layers,
            'format': 'mxnet',
            'multilayers': num_layers != 1,
            'gate_order': None,
        }

        if mode == 'rnn_tanh':
            node_attrs['gate_order'] = [0]
            node_attrs['activations'] = ['tanh'] if not bidirectional else ['tanh', 'tanh']
            RNN.update_node_stat(node, node_attrs)
        elif mode == 'rnn_relu':
            node_attrs['gate_order'] = [0]
            node_attrs['activations'] = ['relu'] if not bidirectional else ['relu', 'relu']
            RNN.update_node_stat(node, node_attrs)
        elif mode == 'gru':
            node_attrs['gate_order'] = [1, 0, 2]
            node_attrs['linear_before_reset'] = 1
            GRU.update_node_stat(node, node_attrs)
        elif mode == 'lstm':
            node_attrs['gate_order'] = [1, 0, 2, 3]
            LSTM.update_node_stat(node, node_attrs)
        else:
            raise Error(
                "Operation RNN with mode '{}' not supported." +
                refer_to_faq_msg(86),
                mode)
        return cls.enabled
