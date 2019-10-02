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

from extensions.ops.elementwise import Mul, Sub, Add, Maximum, Minimum, Div, Greater, GreaterEqual, Equal, Less, LessEqual, Pow, NotEqual, LogicalAnd, LogicalOr
from mo.front.extractor import FrontExtractorOp
from mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs
from mo.graph.graph import Node
from mo.ops.eltwise_n import EltwiseNAdd
from mo.ops.power import Power


class PlusExtractor(FrontExtractorOp):
    op = '_Plus'
    enabled = True

    @staticmethod
    def extract(node: Node):
        Add.update_node_stat(node)
        return __class__.enabled


class BroadcastAddFrontExtractor(FrontExtractorOp):
    op = 'broadcast_add'
    enabled = True

    @staticmethod
    def extract(node):
        Add.update_node_stat(node)
        return __class__.enabled


class BroadcastDivFrontExtractor(FrontExtractorOp):
    op = 'broadcast_div'
    enabled = True

    @staticmethod
    def extract(node):
        Div.update_node_stat(node)
        return __class__.enabled


class BroadcastSubFrontExtractor(FrontExtractorOp):
    op = 'broadcast_sub'
    enabled = True

    @staticmethod
    def extract(node):
        Sub.update_node_stat(node)
        return __class__.enabled


class ElementwiseAddExtractor(FrontExtractorOp):
    op = 'elemwise_add'
    enabled = True

    @staticmethod
    def extract(node: Node):
        Add.update_node_stat(node)
        return __class__.enabled


class ElementWiseSum(FrontExtractorOp):
    op = 'ElementWiseSum'
    enabled = True

    @staticmethod
    def extract(node: Node):
        EltwiseNAdd.update_node_stat(node)
        return __class__.enabled


class AddNExtractor(FrontExtractorOp):
    op = 'add_n'
    enabled = True

    @staticmethod
    def extract(node: Node):
        EltwiseNAdd.update_node_stat(node)
        return __class__.enabled


class ElementwiseMulExtractor(FrontExtractorOp):
    op = 'elemwise_mul'
    enabled = True

    @staticmethod
    def extract(node: Node):
        Mul.update_node_stat(node)
        return __class__.enabled


class BroadcastMulFrontExtractor(FrontExtractorOp):
    op = 'broadcast_mul'
    enabled = True

    @staticmethod
    def extract(node):
        Mul.update_node_stat(node)
        return __class__.enabled


class ElemwiseSubFrontExtractor(FrontExtractorOp):
    op = 'elemwise_sub'
    enabled = True

    @staticmethod
    def extract(node):
        Sub.update_node_stat(node, {})
        return __class__.enabled


class ElemwiseDivFrontExtractor(FrontExtractorOp):
    op = 'elemwise_div'
    enabled = True

    @staticmethod
    def extract(node):
        Div.update_node_stat(node, {})
        return __class__.enabled


class BroadcastMaximumFrontExtractor(FrontExtractorOp):
    op = 'broadcast_maximum'
    enabled = True

    @staticmethod
    def extract(node):
        Maximum.update_node_stat(node)
        return __class__.enabled


class BroadcastMinimumFrontExtractor(FrontExtractorOp):
    op = 'broadcast_minimum'
    enabled = True

    @staticmethod
    def extract(node):
        Minimum.update_node_stat(node)
        return __class__.enabled


class BroadcastGreaterFrontExtractor(FrontExtractorOp):
    op = 'broadcast_greater'
    enabled = True

    @staticmethod
    def extract(node):
        Greater.update_node_stat(node)
        return __class__.enabled


class BroadcastGreaterEqualFrontExtractor(FrontExtractorOp):
    op = 'broadcast_greater_equal'
    enabled = True

    @staticmethod
    def extract(node):
        GreaterEqual.update_node_stat(node)
        return __class__.enabled


class BroadcastEqualFrontExtractor(FrontExtractorOp):
    op = 'broadcast_equal'
    enabled = True

    @staticmethod
    def extract(node):
        Equal.update_node_stat(node)
        return __class__.enabled


class BroadcastNotEqualFrontExtractor(FrontExtractorOp):
    op = 'broadcast_not_equal'
    enabled = True

    @staticmethod
    def extract(node):
        NotEqual.update_node_stat(node)
        return __class__.enabled


class BroadcastLesserFrontExtractor(FrontExtractorOp):
    op = 'broadcast_lesser'
    enabled = True

    @staticmethod
    def extract(node):
        Less.update_node_stat(node)
        return __class__.enabled


class BroadcastLesserEqualFrontExtractor(FrontExtractorOp):
    op = 'broadcast_lesser_equal'
    enabled = True

    @staticmethod
    def extract(node):
        LessEqual.update_node_stat(node)
        return __class__.enabled


class BroadcastPowerFrontExtractor(FrontExtractorOp):
    op = 'broadcast_power'
    enabled = True

    @staticmethod
    def extract(node):
        Pow.update_node_stat(node)
        return __class__.enabled


class BroadcastLogicalAndFrontExtractor(FrontExtractorOp):
    op = 'broadcast_logical_and'
    enabled = True

    @staticmethod
    def extract(node):
        LogicalAnd.update_node_stat(node)
        return __class__.enabled


class BroadcastLogicalOrFrontExtractor(FrontExtractorOp):
    op = 'broadcast_logical_or'
    enabled = True

    @staticmethod
    def extract(node):
        LogicalOr.update_node_stat(node)
        return __class__.enabled


class MaximumFrontExtractor(FrontExtractorOp):
    op = '_maximum'
    enabled = True

    @staticmethod
    def extract(node):
        Maximum.update_node_stat(node)
        return __class__.enabled


class MinimumFrontExtractor(FrontExtractorOp):
    op = '_minimum'
    enabled = True

    @staticmethod
    def extract(node):
        Minimum.update_node_stat(node)
        return __class__.enabled


class PlusScalarFrontExtractor(FrontExtractorOp):
    op = '_plus_scalar'
    enabled = True

    @staticmethod
    def extract(node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        node['scalar'] = np.array([attrs.float('scalar', 0.0)], dtype=np.float32)
        return __class__.enabled


class MinusScalarFrontExtractor(FrontExtractorOp):
    op = '_minus_scalar'
    enabled = True

    @staticmethod
    def extract(node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        node['scalar'] = np.array([attrs.float('scalar', 0.0)])
        return __class__.enabled


class MulScalarFrontExtractor(FrontExtractorOp):
    op = '_mul_scalar'
    enabled = True

    @staticmethod
    def extract(node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        node['scalar'] = np.array([attrs.float('scalar', 1.0)], dtype=np.float32)
        return __class__.enabled


class DivScalarFrontExtractor(FrontExtractorOp):
    op = '_div_scalar'
    enabled = True

    @staticmethod
    def extract(node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        node['scalar'] = attrs.float('scalar', 1.0)
        return __class__.enabled


class GreaterScalarFrontExtractor(FrontExtractorOp):
    op = '_greater_scalar'
    enabled = True

    @staticmethod
    def extract(node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        node['scalar'] = np.array([attrs.float('scalar', 1.0)])
        return __class__.enabled


class GreaterEqualScalarFrontExtractor(FrontExtractorOp):
    op = '_greater_equal_scalar'
    enabled = True

    @staticmethod
    def extract(node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        node['scalar'] = np.array([attrs.float('scalar', 1.0)])
        return __class__.enabled


class EqualScalarFrontExtractor(FrontExtractorOp):
    op = '_equal_scalar'
    enabled = True

    @staticmethod
    def extract(node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        node['scalar'] = np.array([attrs.float('scalar', 1.0)])
        return __class__.enabled


class NotEqualScalarFrontExtractor(FrontExtractorOp):
    op = '_not_equal_scalar'
    enabled = True

    @staticmethod
    def extract(node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        node['scalar'] = np.array([attrs.float('scalar', 1.0)])
        return __class__.enabled


class LesserScalarFrontExtractor(FrontExtractorOp):
    op = '_lesser_scalar'
    enabled = True

    @staticmethod
    def extract(node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        node['scalar'] = np.array([attrs.float('scalar', 1.0)])
        return __class__.enabled


class LesserEqualScalarFrontExtractor(FrontExtractorOp):
    op = '_lesser_equal_scalar'
    enabled = True

    @staticmethod
    def extract(node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        node['scalar'] = np.array([attrs.float('scalar', 1.0)])
        return __class__.enabled


class MinimumScalarFrontExtractor(FrontExtractorOp):
    op = '_minimum_scalar'
    enabled = True

    @staticmethod
    def extract(node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        node['scalar'] = attrs.float('scalar', 1.0)
        return __class__.enabled


class MaximumScalarFrontExtractor(FrontExtractorOp):
    op = '_maximum_scalar'
    enabled = True

    @staticmethod
    def extract(node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        node['scalar'] = attrs.float('scalar', 1.0)
        return __class__.enabled


class ZerosFrontExtractor(FrontExtractorOp):
    op = 'zeros_like'
    enabled = True

    @staticmethod
    def extract(node):
        Power.update_node_stat(node, {'scale': 0})
        return __class__.enabled


class OnesFrontExtractor(FrontExtractorOp):
    op = 'ones_like'
    enabled = True

    @staticmethod
    def extract(node):
        Power.update_node_stat(node, {'scale': 0, 'shift': 1})
        return __class__.enabled
