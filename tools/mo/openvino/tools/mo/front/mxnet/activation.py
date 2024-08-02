# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.activation_ops import SoftPlus, Sigmoid, Tanh, ReLU, \
    Asinh, Acosh, Atanh, SoftSign
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs
from openvino.tools.mo.utils.error import Error
from openvino.tools.mo.utils.utils import refer_to_faq_msg


class ActivationFrontExtractor(FrontExtractorOp):
    op = 'Activation'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        act_type = attrs.str('act_type', 'relu')
        if act_type == 'sigmoid':
            act_class = Sigmoid
        elif act_type == 'tanh':
            act_class = Tanh
        elif act_type == 'relu':
            act_class = ReLU
        elif act_type == 'softrelu':
            act_class = SoftPlus
        elif act_type == 'softsign':
            act_class = SoftSign
        else:
            raise Error(
                "Operation '{}' not supported. Please register it as custom op. " +
                refer_to_faq_msg(86),
                act_type)
        act_class.update_node_stat(node)
        return cls.enabled


class AsinhFrontExtractor(FrontExtractorOp):
    op = 'arcsinh'
    enabled = True

    @classmethod
    def extract(cls, node):
        Asinh.update_node_stat(node)
        return cls.enabled


class AcoshFrontExtractor(FrontExtractorOp):
    op = 'arccosh'
    enabled = True

    @classmethod
    def extract(cls, node):
        Acosh.update_node_stat(node)
        return cls.enabled


class AtanhFrontExtractor(FrontExtractorOp):
    op = 'arctanh'
    enabled = True

    @classmethod
    def extract(cls, node):
        Atanh.update_node_stat(node)
        return cls.enabled
