# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.elementwise import Add, Mul, Sub, Div, Maximum, Minimum, Pow, LogicalAnd, LogicalOr, Equal, \
    GreaterEqual, Greater, Less, LessEqual, NotEqual, FloorMod, BiasAdd, SquaredDifference, Round, Mod
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.tf.extractors.utils import tf_dtype_extractor
from openvino.tools.mo.ops.eltwise_n import EltwiseNAdd
from openvino.tools.mo.ops.power import AttributedPower


class AddExtractor(FrontExtractorOp):
    op = 'Add'
    enabled = True

    @classmethod
    def extract(cls, node):
        Add.update_node_stat(node, {'data_type': tf_dtype_extractor(node.pb.attr["T"].type)})
        return cls.enabled


class AddV2Extractor(FrontExtractorOp):
    op = 'AddV2'
    enabled = True

    @classmethod
    def extract(cls, node):
        Add.update_node_stat(node, {'data_type': tf_dtype_extractor(node.pb.attr["T"].type)})
        return cls.enabled


class AddNExtractor(FrontExtractorOp):
    op = 'AddN'
    enabled = True

    @classmethod
    def extract(cls, node):
        EltwiseNAdd.update_node_stat(node)
        return cls.enabled


class BiasAddExtractor(FrontExtractorOp):
    op = 'BiasAdd'
    enabled = True

    @classmethod
    def extract(cls, node):
        BiasAdd.update_node_stat(node, {'data_type': tf_dtype_extractor(node.pb.attr["T"].type),
                                        'data_format': node.pb.attr["data_format"].s.decode()})
        return cls.enabled


class MulExtractor(FrontExtractorOp):
    op = 'Mul'
    enabled = True

    @classmethod
    def extract(cls, node):
        Mul.update_node_stat(node, {'data_type': tf_dtype_extractor(node.pb.attr["T"].type)})
        return cls.enabled


class SubExtractor(FrontExtractorOp):
    op = 'Sub'
    enabled = True

    @classmethod
    def extract(cls, node):
        Sub.update_node_stat(node, {'data_type': tf_dtype_extractor(node.pb.attr["T"].type)})
        return cls.enabled


class ModExtractor(FrontExtractorOp):
    op = 'Mod'
    enabled = True

    @classmethod
    def extract(cls, node):
        Mod.update_node_stat(node, {'data_type': tf_dtype_extractor(node.pb.attr["T"].type)})
        return cls.enabled


class DivExtractor(FrontExtractorOp):
    op = 'RealDiv'
    enabled = True

    @classmethod
    def extract(cls, node):
        Div.update_node_stat(node, {'data_type': tf_dtype_extractor(node.pb.attr["T"].type)})
        return cls.enabled


class SquaredDifferenceExtractor(FrontExtractorOp):
    op = 'SquaredDifference'
    enabled = True

    @classmethod
    def extract(cls, node):
        SquaredDifference.update_node_stat(node, {'data_type': tf_dtype_extractor(node.pb.attr["T"].type)})
        return cls.enabled


class SqrtExtractor(FrontExtractorOp):
    op = 'Sqrt'
    enabled = True

    @classmethod
    def extract(cls, node):
        AttributedPower.update_node_stat(node, {'power': 0.5})
        return cls.enabled


class RsqrtExtractor(FrontExtractorOp):
    op = 'Rsqrt'
    enabled = True

    @classmethod
    def extract(cls, node):
        AttributedPower.update_node_stat(node, {'power': -0.5})
        return cls.enabled


class SquareExtractor(FrontExtractorOp):
    op = 'Square'
    enabled = True

    @classmethod
    def extract(cls, node):
        data_type = tf_dtype_extractor(node.pb.attr["T"].type)
        AttributedPower.update_node_stat(node, {'power': data_type(2), 'data_type': data_type})
        return cls.enabled


class NegExtractor(FrontExtractorOp):
    op = 'Neg'
    enabled = True

    @classmethod
    def extract(cls, node):
        AttributedPower.update_node_stat(node, {'scale': -1})
        return cls.enabled


class ZerosLike(FrontExtractorOp):
    op = 'ZerosLike'
    enabled = True

    @classmethod
    def extract(cls, node):
        AttributedPower.update_node_stat(node, {'scale': 0})
        return cls.enabled


class MaximumExtractor(FrontExtractorOp):
    op = 'Maximum'
    enabled = True

    @classmethod
    def extract(cls, node):
        Maximum.update_node_stat(node, {'data_type': tf_dtype_extractor(node.pb.attr["T"].type)})
        return cls.enabled


class MinimumExtractor(FrontExtractorOp):
    op = 'Minimum'
    enabled = True

    @classmethod
    def extract(cls, node):
        Minimum.update_node_stat(node, {'data_type': tf_dtype_extractor(node.pb.attr["T"].type)})
        return cls.enabled


class PowExtractor(FrontExtractorOp):
    op = 'Pow'
    enabled = True

    @classmethod
    def extract(cls, node):
        Pow.update_node_stat(node, {'data_type': tf_dtype_extractor(node.pb.attr["T"].type)})
        return cls.enabled


class LogicalAndFrontExtractor(FrontExtractorOp):
    op = 'LogicalAnd'
    enabled = True

    @classmethod
    def extract(cls, node):
        LogicalAnd.update_node_stat(node)
        return cls.enabled


class LogicalOrFrontExtractor(FrontExtractorOp):
    op = 'LogicalOr'
    enabled = True

    @classmethod
    def extract(cls, node):
        LogicalOr.update_node_stat(node)
        return cls.enabled


class EqualExtractor(FrontExtractorOp):
    op = 'Equal'
    enabled = True

    @classmethod
    def extract(cls, node):
        Equal.update_node_stat(node)
        return cls.enabled


class LessEqualExtractor(FrontExtractorOp):
    op = 'LessEqual'
    enabled = True

    @classmethod
    def extract(cls, node):
        LessEqual.update_node_stat(node)
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


class GreaterEqualExtractor(FrontExtractorOp):
    op = 'GreaterEqual'
    enabled = True

    @classmethod
    def extract(cls, node):
        GreaterEqual.update_node_stat(node)
        return cls.enabled


class NotEqualExtractor(FrontExtractorOp):
    op = 'NotEqual'
    enabled = True

    @classmethod
    def extract(cls, node):
        NotEqual.update_node_stat(node)
        return cls.enabled


class FloorModFrontExtractor(FrontExtractorOp):
    op = 'FloorMod'
    enabled = True

    @classmethod
    def extract(cls, node):
        FloorMod.update_node_stat(node)
        return cls.enabled


class RoundExtractor(FrontExtractorOp):
    op = 'Round'
    enabled = True

    @classmethod
    def extract(cls, node):
        Round.update_node_stat(node, {'mode': 'half_to_even'})
        return cls.enabled
