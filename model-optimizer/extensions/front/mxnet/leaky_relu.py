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

from extensions.ops.activation_ops import Elu, LeakyReLU, ReLU
from extensions.ops.prelu import PreluOp
from mo.front.extractor import FrontExtractorOp
from mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs
from mo.utils.error import Error
from mo.utils.utils import refer_to_faq_msg


class LeakyReLUFrontExtractor(FrontExtractorOp):
    op = 'LeakyReLU'
    enabled = True

    @staticmethod
    def extract(node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        act_type = attrs.str('act_type', 'leaky')
        if act_type == 'prelu':
            prelu_attrs = {'channel_shared': 1,
                           'filler_type': 'constant',
                           'filler_value': 0,
                           'min': 0,
                           'max': 1,
                           'mean': 0,
                           'std': 0,
                           'sparse': -1,
                           'variance_norm': "caffe.FillerParameter.FAN_IN"}
            PreluOp.update_node_stat(node, prelu_attrs)
        elif act_type == 'elu':
            alpha = attrs.float('slope', 0.25)
            Elu.update_node_stat(node, {'alpha': alpha})
        elif act_type == 'leaky':
            negative_slope = attrs.float('slope', 0.25)
            if negative_slope == 0:
                ReLU.update_node_stat(node)
            else:
                LeakyReLU.update_node_stat(node, {'negative_slope': negative_slope})
        else:
            raise Error(
                "Operation '{}' not supported. Please register it as custom op. " +
                refer_to_faq_msg(86),
                act_type)

        return LeakyReLUFrontExtractor.enabled
