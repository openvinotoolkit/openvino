# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.activation_ops import *
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.onnx.extractors.utils import onnx_attr


class AbsExtractor(FrontExtractorOp):
    op = 'Abs'
    enabled = True

    @classmethod
    def extract(cls, node):
        Abs.update_node_stat(node)
        return cls.enabled


class AcosExtractor(FrontExtractorOp):
    op = 'Acos'
    enabled = True

    @classmethod
    def extract(cls, node):
        Acos.update_node_stat(node)
        return cls.enabled


class AcoshExtractor(FrontExtractorOp):
    op = 'Acosh'
    enabled = True

    @classmethod
    def extract(cls, node):
        Acosh.update_node_stat(node)
        return cls.enabled


class AsinExtractor(FrontExtractorOp):
    op = 'Asin'
    enabled = True

    @classmethod
    def extract(cls, node):
        Asin.update_node_stat(node)
        return cls.enabled


class AsinhExtractor(FrontExtractorOp):
    op = 'Asinh'
    enabled = True

    @classmethod
    def extract(cls, node):
        Asinh.update_node_stat(node)
        return cls.enabled


class AtanExtractor(FrontExtractorOp):
    op = 'Atan'
    enabled = True

    @classmethod
    def extract(cls, node):
        Atan.update_node_stat(node)
        return cls.enabled


class AtanhExtractor(FrontExtractorOp):
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


class CosExtractor(FrontExtractorOp):
    op = 'Cos'
    enabled = True

    @classmethod
    def extract(cls, node):
        Cos.update_node_stat(node)
        return cls.enabled


class CoshExtractor(FrontExtractorOp):
    op = 'Cosh'
    enabled = True

    @classmethod
    def extract(cls, node):
        Cosh.update_node_stat(node)
        return cls.enabled


class EluExtractor(FrontExtractorOp):
    op = 'Elu'
    enabled = True

    @classmethod
    def extract(cls, node):
        alpha = onnx_attr(node, 'alpha', 'f', default=1.0)
        Elu.update_node_stat(node, {'alpha': alpha})
        return EluExtractor.enabled


class ErfExtractor(FrontExtractorOp):
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


class FloorExtractor(FrontExtractorOp):
    op = 'Floor'
    enabled = True

    @classmethod
    def extract(cls, node):
        Floor.update_node_stat(node)
        return cls.enabled


class ThresholdedReluExtractor(FrontExtractorOp):
    op = 'ThresholdedRelu'
    enabled = True

    @classmethod
    def extract(cls, node):
        alpha = onnx_attr(node, 'alpha', 'f', default=1.0)
        ThresholdedRelu.update_node_stat(node, {'alpha': alpha})
        return cls.enabled


class LeakyReLUExtractor(FrontExtractorOp):
    op = 'LeakyRelu'
    enabled = True

    @classmethod
    def extract(cls, node):
        negative_slope = onnx_attr(node, 'alpha', 'f', default=1.0)
        if negative_slope == 0:
            ReLU.update_node_stat(node)
        else:
            LeakyReLU.update_node_stat(node, {'negative_slope': negative_slope})
        return cls.enabled


class LogExtractor(FrontExtractorOp):
    op = 'Log'
    enabled = True

    @classmethod
    def extract(cls, node):
        Log.update_node_stat(node)
        return cls.enabled


class NotExtractor(FrontExtractorOp):
    op = 'Not'
    enabled = True

    @classmethod
    def extract(cls, node):
        LogicalNot.update_node_stat(node)
        return cls.enabled


class ReLUExtractor(FrontExtractorOp):
    op = 'Relu'
    enabled = True

    @classmethod
    def extract(cls, node):
        ReLU.update_node_stat(node)
        return cls.enabled


class SigmoidExtractor(FrontExtractorOp):
    op = 'Sigmoid'
    enabled = True

    @classmethod
    def extract(cls, node):
        Sigmoid.update_node_stat(node)
        return cls.enabled


class SignExtractor(FrontExtractorOp):
    op = 'Sign'
    enabled = True

    @classmethod
    def extract(cls, node):
        Sign.update_node_stat(node)
        return cls.enabled


class SinExtractor(FrontExtractorOp):
    op = 'Sin'
    enabled = True

    @classmethod
    def extract(cls, node):
        Sin.update_node_stat(node)
        return cls.enabled


class SinhExtractor(FrontExtractorOp):
    op = 'Sinh'
    enabled = True

    @classmethod
    def extract(cls, node):
        Sinh.update_node_stat(node)
        return cls.enabled


class TanExtractor(FrontExtractorOp):
    op = 'Tan'
    enabled = True

    @classmethod
    def extract(cls, node):
        Tan.update_node_stat(node)
        return cls.enabled


class TanhExtractor(FrontExtractorOp):
    op = 'Tanh'
    enabled = True

    @classmethod
    def extract(cls, node):
        Tanh.update_node_stat(node)
        return cls.enabled


class SoftSignExtractor(FrontExtractorOp):
    op = 'Softsign'
    enabled = True

    @classmethod
    def extract(cls, node):
        SoftSign.update_node_stat(node, {})
        return cls.enabled
