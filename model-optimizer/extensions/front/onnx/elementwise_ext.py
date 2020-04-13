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
import numpy as np

from extensions.ops.elementwise import Add, Sub, Mul, Div, Pow, Less, Equal, Greater, \
    LogicalAnd, LogicalOr, LogicalXor
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr
from mo.graph.graph import Node
from mo.ops.eltwise_n import EltwiseNAdd, EltwiseNMax
from mo.ops.power import AttributedPower


class AddFrontExtractor(FrontExtractorOp):
    op = 'Add'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        axis = onnx_attr(node, 'axis', 'i', default=None)
        Add.update_node_stat(node, {'axis': axis})
        return cls.enabled


class SubFrontExtractor(FrontExtractorOp):
    op = 'Sub'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        axis = onnx_attr(node, 'axis', 'i', default=None)
        Sub.update_node_stat(node, {'axis': axis})
        return cls.enabled


class MulFrontExtractor(FrontExtractorOp):
    op = 'Mul'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        axis = onnx_attr(node, 'axis', 'i', default=None)
        Mul.update_node_stat(node, {'axis': axis})
        return cls.enabled


class DivFrontExtractor(FrontExtractorOp):
    op = 'Div'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        axis = onnx_attr(node, 'axis', 'i', default=None)
        Div.update_node_stat(node, {'axis': axis})
        return cls.enabled


class SumFrontExtractor(FrontExtractorOp):
    op = 'Sum'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        axis = onnx_attr(node, 'axis', 'i', default=None)
        EltwiseNAdd.update_node_stat(node, {'axis': axis})
        return cls.enabled


class PowFrontExtractor(FrontExtractorOp):
    op = 'Pow'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        Pow.update_node_stat(node)
        return cls.enabled


class NegFrontExtractor(FrontExtractorOp):
    op = 'Neg'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        AttributedPower.update_node_stat(node, {'scale': -1})
        return cls.enabled


class SqrtExtractor(FrontExtractorOp):
    op = 'Sqrt'
    enabled = True

    @classmethod
    def extract(cls, node):
        AttributedPower.update_node_stat(node, {'power': 0.5})
        return cls.enabled


class ScaleFrontExtractor(FrontExtractorOp):
    op = 'Scale'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        scale = onnx_attr(node, 'scale', 'f', default=np.array(1.0), dst_type=lambda x: np.array(x))
        AttributedPower.update_node_stat(node, {'scale': scale})
        return cls.enabled


class MaxExtractor(FrontExtractorOp):
    op = 'Max'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        EltwiseNMax.update_node_stat(node)
        return cls.enabled


class EqualExtractor(FrontExtractorOp):
    op = 'Equal'
    enabled = True

    @classmethod
    def extract(cls, node):
        Equal.update_node_stat(node)
        return cls.enabled


class LessExtractor(FrontExtractorOp):
    op = 'Less'
    enabled = True

    @classmethod
    def extract(cls, node):
        Less.update_node_stat(node)
        return cls.enabled


class GreaterExtractor(FrontExtractorOp):
    op = 'Greater'
    enabled = True

    @classmethod
    def extract(cls, node):
        Greater.update_node_stat(node)
        return cls.enabled


class AndExtractor(FrontExtractorOp):
    op = 'And'
    enabled = True

    @classmethod
    def extract(cls, node):
        LogicalAnd.update_node_stat(node)
        return cls.enabled


class OrExtractor(FrontExtractorOp):
    op = 'Or'
    enabled = True

    @classmethod
    def extract(cls, node):
        LogicalOr.update_node_stat(node)
        return cls.enabled


class XorExtractor(FrontExtractorOp):
    op = 'Xor'
    enabled = True

    @classmethod
    def extract(cls, node):
        LogicalXor.update_node_stat(node)
        return cls.enabled
