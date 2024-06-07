# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.activation_ops import Elu, LeakyReLU, ReLU
from openvino.tools.mo.ops.gelu import GeLUOP
from openvino.tools.mo.ops.prelu import PReLU
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs
from openvino.tools.mo.utils.error import Error
from openvino.tools.mo.utils.utils import refer_to_faq_msg


class LeakyReLUFrontExtractor(FrontExtractorOp):
    op = 'LeakyReLU'
    enabled = True

    @classmethod
    def extract(cls, node):
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
            PReLU.update_node_stat(node, prelu_attrs)
        elif act_type == 'elu':
            alpha = attrs.float('slope', 0.25)
            Elu.update_node_stat(node, {'alpha': alpha})
        elif act_type == 'leaky':
            negative_slope = attrs.float('slope', 0.25)
            if negative_slope == 0:
                ReLU.update_node_stat(node)
            else:
                LeakyReLU.update_node_stat(node, {'negative_slope': negative_slope})
        elif act_type == 'gelu':
            GeLUOP.update_node_stat(node, {'approximation_mode': 'erf'})
        else:
            raise Error(
                "Operation '{}' not supported. Please register it as custom op. " +
                refer_to_faq_msg(86),
                act_type)

        return LeakyReLUFrontExtractor.enabled
