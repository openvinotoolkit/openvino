"""
 Copyright (c) 2018 Intel Corporation

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

from mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs
from mo.front.extractor import FrontExtractorOp
from extensions.ops.lstm_sequence import LSTMSequence
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

        node_attrs = {
            'batch_dim': 1,
            'sequence_dim': 0,
            'blobs_wrb': False,
            'hidden_size': state_size,
            'has_num_directions': bidirectional,
            'format': 'mxnet',
        }

        if bidirectional:
            raise Error(
                "Operation RNN with bidirectional not supported. num_directions = 1 is supported only " +
                refer_to_faq_msg(86))

        if num_layers > 1:
            raise Error(
                "Operation RNN with num_layers more then one not supported. num_layers = 1 is supported only " +
                refer_to_faq_msg(86))

        if mode == 'lstm':
            LSTMSequence.update_node_stat(node, node_attrs)
        else:
            raise Error(
                "Operation RNN with mode '{}' not supported. Please register RNN as custom op. " +
                refer_to_faq_msg(86),
                mode)
        return __class__.enabled
