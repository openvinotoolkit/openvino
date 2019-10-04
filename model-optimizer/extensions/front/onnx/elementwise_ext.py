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
import numpy as np

from extensions.ops.elementwise import Add, Mul, Pow, Less, Equal, Greater, LogicalAnd, LogicalOr
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr
from mo.graph.graph import Node
from mo.ops.eltwise_n import EltwiseNAdd, EltwiseNMax
from mo.ops.power import Power


class AddFrontExtractor(FrontExtractorOp):
    op = 'Add'
    enabled = True

    @staticmethod
    def extract(node: Node):
        axis = onnx_attr(node, 'axis', 'i', default=None)
        Add.update_node_stat(node, {'axis': axis})
        return __class__.enabled


class MulFrontExtractor(FrontExtractorOp):
    op = 'Mul'
    enabled = True

    @staticmethod
    def extract(node: Node):
        axis = onnx_attr(node, 'axis', 'i', default=None)
        Mul.update_node_stat(node, {'axis': axis})
        return __class__.enabled


class SumFrontExtractor(FrontExtractorOp):
    op = 'Sum'
    enabled = True

    @staticmethod
    def extract(node: Node):
        axis = onnx_attr(node, 'axis', 'i', default=None)
        EltwiseNAdd.update_node_stat(node, {'axis': axis})
        return __class__.enabled


class PowFrontExtractor(FrontExtractorOp):
    op = 'Pow'
    enabled = True

    @staticmethod
    def extract(node: Node):
        Pow.update_node_stat(node)
        return __class__.enabled


class NegFrontExtractor(FrontExtractorOp):
    op = 'Neg'
    enabled = True

    @staticmethod
    def extract(node: Node):
        Power.update_node_stat(node, {'scale': -1})
        return __class__.enabled


class SqrtExtractor(FrontExtractorOp):
    op = 'Sqrt'
    enabled = True

    @staticmethod
    def extract(node):
        Power.update_node_stat(node, {'power': 0.5})
        return __class__.enabled


class ScaleFrontExtractor(FrontExtractorOp):
    op = 'Scale'
    enabled = True

    @staticmethod
    def extract(node: Node):
        scale = onnx_attr(node, 'scale', 'f', default=np.array(1.0), dst_type=lambda x: np.array(x))
        Power.update_node_stat(node, {'scale': scale})
        return __class__.enabled


class MaxExtractor(FrontExtractorOp):
    op = 'Max'
    enabled = True

    @staticmethod
    def extract(node: Node):
        EltwiseNMax.update_node_stat(node)
        return __class__.enabled


class EqualExtractor(FrontExtractorOp):
    op = 'Equal'
    enabled = True

    @staticmethod
    def extract(node):
        Equal.update_node_stat(node)
        return __class__.enabled


class LessExtractor(FrontExtractorOp):
    op = 'Less'
    enabled = True

    @staticmethod
    def extract(node):
        Less.update_node_stat(node)
        return __class__.enabled


class GreaterExtractor(FrontExtractorOp):
    op = 'Greater'
    enabled = True

    @staticmethod
    def extract(node):
        Greater.update_node_stat(node)
        return __class__.enabled


class AndExtractor(FrontExtractorOp):
    op = 'And'
    enabled = True

    @staticmethod
    def extract(node):
        LogicalAnd.update_node_stat(node)
        return __class__.enabled


class OrExtractor(FrontExtractorOp):
    op = 'Or'
    enabled = True

    @staticmethod
    def extract(node):
        LogicalOr.update_node_stat(node)
        return __class__.enabled
