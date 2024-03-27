# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.ops.elementwise import Mul, Sub, Add, Maximum, Minimum, Div, Greater, GreaterEqual, Equal, Less, \
    LessEqual, Pow, NotEqual, LogicalAnd, LogicalOr, Round
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.ops.eltwise_n import EltwiseNAdd
from openvino.tools.mo.ops.power import AttributedPower


class PlusExtractor(FrontExtractorOp):
    op = '_Plus'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        Add.update_node_stat(node)
        return cls.enabled


class BroadcastAddFrontExtractor(FrontExtractorOp):
    op = 'broadcast_add'
    enabled = True

    @classmethod
    def extract(cls, node):
        Add.update_node_stat(node)
        return cls.enabled


class BroadcastDivFrontExtractor(FrontExtractorOp):
    op = 'broadcast_div'
    enabled = True

    @classmethod
    def extract(cls, node):
        Div.update_node_stat(node)
        return cls.enabled


class BroadcastSubFrontExtractor(FrontExtractorOp):
    op = 'broadcast_sub'
    enabled = True

    @classmethod
    def extract(cls, node):
        Sub.update_node_stat(node)
        return cls.enabled


class ElementwiseAddExtractor(FrontExtractorOp):
    op = 'elemwise_add'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        Add.update_node_stat(node)
        return cls.enabled


class ElementWiseSum(FrontExtractorOp):
    op = 'ElementWiseSum'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        EltwiseNAdd.update_node_stat(node)
        return cls.enabled


class AddNExtractor(FrontExtractorOp):
    op = 'add_n'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        EltwiseNAdd.update_node_stat(node)
        return cls.enabled


class ElementwiseMulExtractor(FrontExtractorOp):
    op = 'elemwise_mul'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        Mul.update_node_stat(node)
        return cls.enabled


class BroadcastMulFrontExtractor(FrontExtractorOp):
    op = 'broadcast_mul'
    enabled = True

    @classmethod
    def extract(cls, node):
        Mul.update_node_stat(node)
        return cls.enabled


class ElemwiseSubFrontExtractor(FrontExtractorOp):
    op = 'elemwise_sub'
    enabled = True

    @classmethod
    def extract(cls, node):
        Sub.update_node_stat(node, {})
        return cls.enabled


class ElemwiseDivFrontExtractor(FrontExtractorOp):
    op = 'elemwise_div'
    enabled = True

    @classmethod
    def extract(cls, node):
        Div.update_node_stat(node, {})
        return cls.enabled


class BroadcastMaximumFrontExtractor(FrontExtractorOp):
    op = 'broadcast_maximum'
    enabled = True

    @classmethod
    def extract(cls, node):
        Maximum.update_node_stat(node)
        return cls.enabled


class BroadcastMinimumFrontExtractor(FrontExtractorOp):
    op = 'broadcast_minimum'
    enabled = True

    @classmethod
    def extract(cls, node):
        Minimum.update_node_stat(node)
        return cls.enabled


class BroadcastGreaterFrontExtractor(FrontExtractorOp):
    op = 'broadcast_greater'
    enabled = True

    @classmethod
    def extract(cls, node):
        Greater.update_node_stat(node)
        return cls.enabled


class BroadcastGreaterEqualFrontExtractor(FrontExtractorOp):
    op = 'broadcast_greater_equal'
    enabled = True

    @classmethod
    def extract(cls, node):
        GreaterEqual.update_node_stat(node)
        return cls.enabled


class BroadcastEqualFrontExtractor(FrontExtractorOp):
    op = 'broadcast_equal'
    enabled = True

    @classmethod
    def extract(cls, node):
        Equal.update_node_stat(node)
        return cls.enabled


class BroadcastNotEqualFrontExtractor(FrontExtractorOp):
    op = 'broadcast_not_equal'
    enabled = True

    @classmethod
    def extract(cls, node):
        NotEqual.update_node_stat(node)
        return cls.enabled


class BroadcastLesserFrontExtractor(FrontExtractorOp):
    op = 'broadcast_lesser'
    enabled = True

    @classmethod
    def extract(cls, node):
        Less.update_node_stat(node)
        return cls.enabled


class BroadcastLesserEqualFrontExtractor(FrontExtractorOp):
    op = 'broadcast_lesser_equal'
    enabled = True

    @classmethod
    def extract(cls, node):
        LessEqual.update_node_stat(node)
        return cls.enabled


class BroadcastPowerFrontExtractor(FrontExtractorOp):
    op = 'broadcast_power'
    enabled = True

    @classmethod
    def extract(cls, node):
        Pow.update_node_stat(node)
        return cls.enabled


class BroadcastLogicalAndFrontExtractor(FrontExtractorOp):
    op = 'broadcast_logical_and'
    enabled = True

    @classmethod
    def extract(cls, node):
        LogicalAnd.update_node_stat(node)
        return cls.enabled


class BroadcastLogicalOrFrontExtractor(FrontExtractorOp):
    op = 'broadcast_logical_or'
    enabled = True

    @classmethod
    def extract(cls, node):
        LogicalOr.update_node_stat(node)
        return cls.enabled


class MaximumFrontExtractor(FrontExtractorOp):
    op = '_maximum'
    enabled = True

    @classmethod
    def extract(cls, node):
        Maximum.update_node_stat(node)
        return cls.enabled


class MinimumFrontExtractor(FrontExtractorOp):
    op = '_minimum'
    enabled = True

    @classmethod
    def extract(cls, node):
        Minimum.update_node_stat(node)
        return cls.enabled


class PlusScalarFrontExtractor(FrontExtractorOp):
    op = '_plus_scalar'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        node['scalar'] = mo_array([attrs.float('scalar', 0.0)], dtype=np.float32)
        return cls.enabled


class MinusScalarFrontExtractor(FrontExtractorOp):
    op = '_minus_scalar'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        node['scalar'] = mo_array([attrs.float('scalar', 0.0)])
        return cls.enabled


class MulScalarFrontExtractor(FrontExtractorOp):
    op = '_mul_scalar'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        node['scalar'] = mo_array([attrs.float('scalar', 1.0)], dtype=np.float32)
        return cls.enabled


class DivScalarFrontExtractor(FrontExtractorOp):
    op = '_div_scalar'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        node['scalar'] = attrs.float('scalar', 1.0)
        return cls.enabled


class GreaterScalarFrontExtractor(FrontExtractorOp):
    op = '_greater_scalar'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        node['scalar'] = mo_array([attrs.float('scalar', 1.0)])
        return cls.enabled


class GreaterEqualScalarFrontExtractor(FrontExtractorOp):
    op = '_greater_equal_scalar'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        node['scalar'] = mo_array([attrs.float('scalar', 1.0)])
        return cls.enabled


class EqualScalarFrontExtractor(FrontExtractorOp):
    op = '_equal_scalar'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        node['scalar'] = mo_array([attrs.float('scalar', 1.0)])
        return cls.enabled


class NotEqualScalarFrontExtractor(FrontExtractorOp):
    op = '_not_equal_scalar'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        node['scalar'] = mo_array([attrs.float('scalar', 1.0)])
        return cls.enabled


class LesserScalarFrontExtractor(FrontExtractorOp):
    op = '_lesser_scalar'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        node['scalar'] = mo_array([attrs.float('scalar', 1.0)])
        return cls.enabled


class LesserEqualScalarFrontExtractor(FrontExtractorOp):
    op = '_lesser_equal_scalar'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        node['scalar'] = mo_array([attrs.float('scalar', 1.0)])
        return cls.enabled


class MinimumScalarFrontExtractor(FrontExtractorOp):
    op = '_minimum_scalar'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        node['scalar'] = attrs.float('scalar', 1.0)
        return cls.enabled


class MaximumScalarFrontExtractor(FrontExtractorOp):
    op = '_maximum_scalar'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        node['scalar'] = attrs.float('scalar', 1.0)
        return cls.enabled


class ZerosFrontExtractor(FrontExtractorOp):
    op = 'zeros_like'
    enabled = True

    @classmethod
    def extract(cls, node):
        AttributedPower.update_node_stat(node, {'scale': 0})
        return cls.enabled


class OnesFrontExtractor(FrontExtractorOp):
    op = 'ones_like'
    enabled = True

    @classmethod
    def extract(cls, node):
        AttributedPower.update_node_stat(node, {'scale': 0, 'shift': 1})
        return cls.enabled


class RoundExtractor(FrontExtractorOp):
    op = 'round'
    enabled = True

    @classmethod
    def extract(cls, node):
        Round.update_node_stat(node, {'mode': 'half_away_from_zero'})
        return cls.enabled
