"""
 Copyright (C) 2018-2020 Intel Corporation

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
from extensions.ops.activation_ops import Abs, Elu, Erf, Exp, ReLU, LeakyReLU, LogicalNot, ReLU6, Sigmoid, \
    Sin, Sinh, Cos, Cosh, Tan, Tanh, Ceiling, Atanh, Acosh, Asinh
from mo.front.extractor import FrontExtractorOp


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
