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

from mo.ops.activation import Activation
from mo.front.extractor import FrontExtractorOp
from mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs
from mo.ops.relu import ReLU
from mo.utils.error import Error
from mo.utils.utils import refer_to_faq_msg


class ActivationFrontExtractor(FrontExtractorOp):
    op = 'Activation'
    enabled = True

    @staticmethod
    def extract(node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        act_type = attrs.str('act_type', 'relu')
        if act_type == 'sigmoid' or act_type == 'tanh':
            Activation.update_node_stat(node, {'operation': act_type})
        elif act_type == 'relu':
            ReLU.update_node_stat(node)
        else:
            raise Error(
                "Operation '{}' not supported. Please register it as custom op. " +
                refer_to_faq_msg(86),
                act_type)

        return ActivationFrontExtractor.enabled
