# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.activation_ops import *
from openvino.tools.mo.front.extractor import FrontExtractorOp


class AbsExtractor(FrontExtractorOp):
    op = 'Abs'
    enabled = True

    @classmethod
    def extract(cls, node):
        Abs.update_node_stat(node)
        return cls.enabled


class EluFrontExtractor(FrontExtractorOp):
    op = 'Elu'
    enabled = True

    @classmethod
    def extract(cls, node):
        Elu.update_node_stat(node)
        return cls.enabled


class ErfFrontExtractor(FrontExtractorOp):
    op = 'Erf'
    enabled = True

    @classmethod
    def extract(cls, node):
        Erf.update_node_stat(node)
        return cls.enabled


class ExpExtractor(FrontExtractorOp):
    op = 'Exp'
    enabled = True

    @classmethod
    def extract(cls, node):
        Exp.update_node_stat(node)
        return cls.enabled


class LeakyReLUFrontExtractor(FrontExtractorOp):
    op = 'LeakyRelu'
    enabled = True

    @classmethod
    def extract(cls, node):
        negative_slope = node.pb.attr['alpha'].f
        if negative_slope == 0:
            ReLU.update_node_stat(node)
        else:
            LeakyReLU.update_node_stat(node, {'negative_slope': negative_slope})
        return cls.enabled


class LogicalNotFrontExtractor(FrontExtractorOp):
    op = 'LogicalNot'
    enabled = True

    @classmethod
    def extract(cls, node):
        LogicalNot.update_node_stat(node)
        return cls.enabled


class Relu6FrontExtractor(FrontExtractorOp):
    op = 'Relu6'
    enabled = True

    @classmethod
    def extract(cls, node):
        ReLU6.update_node_stat(node)
        return cls.enabled


class ReluFrontExtractor(FrontExtractorOp):
    op = 'Relu'
    enabled = True

    @classmethod
    def extract(cls, node):
        ReLU.update_node_stat(node)
        return cls.enabled


class SigmoidFrontExtractor(FrontExtractorOp):
    op = 'Sigmoid'
    enabled = True

    @classmethod
    def extract(cls, node):
        Sigmoid.update_node_stat(node)
        return cls.enabled


class CosFrontExtractor(FrontExtractorOp):
    op = 'Cos'
    enabled = True

    @classmethod
    def extract(cls, node):
        Cos.update_node_stat(node)
        return cls.enabled


class CoshFrontExtractor(FrontExtractorOp):
    op = 'Cosh'
    enabled = True

    @classmethod
    def extract(cls, node):
        Cosh.update_node_stat(node)
        return cls.enabled


class AcoshFrontExtractor(FrontExtractorOp):
    op = 'Acosh'
    enabled = True

    @classmethod
    def extract(cls, node):
        Acosh.update_node_stat(node)
        return cls.enabled


class SinFrontExtractor(FrontExtractorOp):
    op = 'Sin'
    enabled = True

    @classmethod
    def extract(cls, node):
        Sin.update_node_stat(node)
        return cls.enabled


class SinhFrontExtractor(FrontExtractorOp):
    op = 'Sinh'
    enabled = True

    @classmethod
    def extract(cls, node):
        Sinh.update_node_stat(node)
        return cls.enabled


class AsinhFrontExtractor(FrontExtractorOp):
    op = 'Asinh'
    enabled = True

    @classmethod
    def extract(cls, node):
        Asinh.update_node_stat(node)
        return cls.enabled


class TanFrontExtractor(FrontExtractorOp):
    op = 'Tan'
    enabled = True

    @classmethod
    def extract(cls, node):
        Tan.update_node_stat(node)
        return cls.enabled


class TanhFrontExtractor(FrontExtractorOp):
    op = 'Tanh'
    enabled = True

    @classmethod
    def extract(cls, node):
        Tanh.update_node_stat(node)
        return cls.enabled


class AtanhFrontExtractor(FrontExtractorOp):
    op = 'Atanh'
    enabled = True

    @classmethod
    def extract(cls, node):
        Atanh.update_node_stat(node)
        return cls.enabled


class CeilExtractor(FrontExtractorOp):
    op = 'Ceil'
    enabled = True

    @classmethod
    def extract(cls, node):
        Ceiling.update_node_stat(node)
        return cls.enabled


class MishExtractor(FrontExtractorOp):
    op = 'Mish'
    enabled = True

    @classmethod
    def extract(cls, node):
        Mish.update_node_stat(node)
        return cls.enabled


class LogExtractor(FrontExtractorOp):
    op = 'Log'
    enabled = True

    @classmethod
    def extract(cls, node):
        Log.update_node_stat(node)
        return cls.enabled


class AsinExtractor(FrontExtractorOp):
    op = 'Asin'
    enabled = True

    @classmethod
    def extract(cls, node):
        Asin.update_node_stat(node)
        return cls.enabled


class AcosExtractor(FrontExtractorOp):
    op = 'Acos'
    enabled = True

    @classmethod
    def extract(cls, node):
        Acos.update_node_stat(node)
        return cls.enabled


class AtanExtractor(FrontExtractorOp):
    op = 'Atan'
    enabled = True

    @classmethod
    def extract(cls, node):
        Atan.update_node_stat(node)
        return cls.enabled


class SoftSignExtractor(FrontExtractorOp):
    op = 'Softsign'
    enabled = True

    @classmethod
    def extract(cls, node):
        SoftSign.update_node_stat(node, {})
        return cls.enabled
