# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from extensions.ops.elementwise import Add, Sub, Mul, Div, Pow, Less, Equal, Greater, LogicalAnd, LogicalOr, LogicalXor, \
    Round, GreaterEqual, LessEqual
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr
from mo.graph.graph import Node
from mo.ops.eltwise_n import EltwiseNAdd, EltwiseNMax, EltwiseNMin
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


class MinExtractor(FrontExtractorOp):
    op = 'Min'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        EltwiseNMin.update_node_stat(node)
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


class GreaterOrEqualExtractor(FrontExtractorOp):
    op = 'GreaterOrEqual'
    enabled = True

    @classmethod
    def extract(cls, node):
        GreaterEqual.update_node_stat(node)
        return cls.enabled


class LessOrEqualExtractor(FrontExtractorOp):
    op = 'LessOrEqual'
    enabled = True

    @classmethod
    def extract(cls, node):
        LessEqual.update_node_stat(node)
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


class RoundFrontExtractor(FrontExtractorOp):
    op = 'Round'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        Round.update_node_stat(node, {'mode': 'half_to_even'})
        return cls.enabled
