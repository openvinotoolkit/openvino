"""
 Copyright (c) 2018-2019 Intel Corporation

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
from extensions.ops.GRU import GRU
from extensions.ops.LSTM import LSTM
from extensions.ops.RNN import RNN
from mo.front.extractor import FrontExtractorOp
from mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs
from mo.utils.error import Error
from mo.utils.utils import refer_to_faq_msg


class RNNFrontExtractor(FrontExtractorOp):
    op = 'RNN'
    enabled = True

    @staticmethod
    def extract(node):
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
            'gate_order':  None,
        }

        if mode == 'rnn_tanh':
            node_attrs['gate_order'] = [0]
            node_attrs['activations'] = ['tanh']
            RNN.update_node_stat(node, node_attrs)
        elif mode == 'rnn_relu':
            node_attrs['gate_order'] = [0]
            node_attrs['activations'] = ['relu']
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
        return __class__.enabled
