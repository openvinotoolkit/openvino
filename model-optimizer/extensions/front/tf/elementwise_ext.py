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
import logging as log

from extensions.ops.elementwise import Add, Mul, Sub, Div, Maximum, Minimum, Pow, LogicalAnd, LogicalOr, Equal, \
    GreaterEqual, Greater, Less, LessEqual, NotEqual, BiasAdd
from mo.front.extractor import FrontExtractorOp
from mo.front.tf.extractors.utils import tf_dtype_extractor
from mo.ops.eltwise_n import EltwiseNAdd
from mo.ops.power import Power


class AddExtractor(FrontExtractorOp):
    op = 'Add'
    enabled = True

    @staticmethod
    def extract(node):
        Add.update_node_stat(node, {'data_type': tf_dtype_extractor(node.pb.attr["T"].type)})
        return __class__.enabled


class AddV2Extractor(FrontExtractorOp):
    op = 'AddV2'
    enabled = True

    @staticmethod
    def extract(node):
        Add.update_node_stat(node, {'data_type': tf_dtype_extractor(node.pb.attr["T"].type)})
        return __class__.enabled


class AddNExtractor(FrontExtractorOp):
    op = 'AddN'
    enabled = True

    @staticmethod
    def extract(node):
        EltwiseNAdd.update_node_stat(node)
        return __class__.enabled


class BiasAddExtractor(FrontExtractorOp):
    op = 'BiasAdd'
    enabled = True

    @staticmethod
    def extract(node):
        BiasAdd.update_node_stat(node, {'data_type': tf_dtype_extractor(node.pb.attr["T"].type),
                                        'data_format': node.pb.attr["data_format"].s.decode()})
        return __class__.enabled


class MulExtractor(FrontExtractorOp):
    op = 'Mul'
    enabled = True

    @staticmethod
    def extract(node):
        Mul.update_node_stat(node, {'data_type': tf_dtype_extractor(node.pb.attr["T"].type)})
        return __class__.enabled


class SubExtractor(FrontExtractorOp):
    op = 'Sub'
    enabled = True

    @staticmethod
    def extract(node):
        Sub.update_node_stat(node, {'data_type': tf_dtype_extractor(node.pb.attr["T"].type)})
        return __class__.enabled


class DivExtractor(FrontExtractorOp):
    op = 'RealDiv'
    enabled = True

    @staticmethod
    def extract(node):
        Div.update_node_stat(node, {'data_type': tf_dtype_extractor(node.pb.attr["T"].type)})
        return __class__.enabled


class SqrtExtractor(FrontExtractorOp):
    op = 'Sqrt'
    enabled = True

    @staticmethod
    def extract(node):
        Power.update_node_stat(node, {'power': 0.5})
        return __class__.enabled


class RsqrtExtractor(FrontExtractorOp):
    op = 'Rsqrt'
    enabled = True

    @staticmethod
    def extract(node):
        Power.update_node_stat(node, {'power': -0.5})
        return __class__.enabled


class SquareExtractor(FrontExtractorOp):
    op = 'Square'
    enabled = True

    @staticmethod
    def extract(node):
        # update the attributes of the node
        Power.update_node_stat(node, {'power': 2})
        return __class__.enabled


class NegExtractor(FrontExtractorOp):
    op = 'Neg'
    enabled = True

    @staticmethod
    def extract(node):
        Power.update_node_stat(node, {'scale': -1})
        return __class__.enabled


class ZerosLike(FrontExtractorOp):
    op = 'ZerosLike'
    enabled = True

    @staticmethod
    def extract(node):
        Power.update_node_stat(node, {'scale': 0})
        return __class__.enabled


class MaximumExtractor(FrontExtractorOp):
    op = 'Maximum'
    enabled = True

    @staticmethod
    def extract(node):
        Maximum.update_node_stat(node, {'data_type': tf_dtype_extractor(node.pb.attr["T"].type)})
        return __class__.enabled


class MinimumExtractor(FrontExtractorOp):
    op = 'Minimum'
    enabled = True

    @staticmethod
    def extract(node):
        Minimum.update_node_stat(node, {'data_type': tf_dtype_extractor(node.pb.attr["T"].type)})
        return __class__.enabled


class PowExtractor(FrontExtractorOp):
    op = 'Pow'
    enabled = True

    @staticmethod
    def extract(node):
        Pow.update_node_stat(node, {'data_type': tf_dtype_extractor(node.pb.attr["T"].type)})
        return __class__.enabled


class LogicalAndFrontExtractor(FrontExtractorOp):
    op = 'LogicalAnd'
    enabled = True

    @staticmethod
    def extract(node):
        LogicalAnd.update_node_stat(node)
        return __class__.enabled


class LogicalOrFrontExtractor(FrontExtractorOp):
    op = 'LogicalOr'
    enabled = True

    @staticmethod
    def extract(node):
        LogicalOr.update_node_stat(node)
        return __class__.enabled


class EqualExtractor(FrontExtractorOp):
    op = 'Equal'
    enabled = True

    @staticmethod
    def extract(node):
        Equal.update_node_stat(node)
        return __class__.enabled


class LessEqualExtractor(FrontExtractorOp):
    op = 'LessEqual'
    enabled = True

    @staticmethod
    def extract(node):
        LessEqual.update_node_stat(node)
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


class GreaterEqualExtractor(FrontExtractorOp):
    op = 'GreaterEqual'
    enabled = True

    @staticmethod
    def extract(node):
        GreaterEqual.update_node_stat(node)
        return __class__.enabled


class NotEqualExtractor(FrontExtractorOp):
    op = 'NotEqual'
    enabled = True

    @staticmethod
    def extract(node):
        NotEqual.update_node_stat(node)
        return __class__.enabled
