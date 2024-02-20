# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.elementwise import Div, Greater, GreaterEqual, Equal, NotEqual, Sub, Mul, Add, Less, LessEqual, Minimum, Maximum
from openvino.tools.mo.front.common.replacement import FrontReplacementOp
from openvino.tools.mo.front.mxnet.extractors.utils import scalar_ops_replacer
from openvino.tools.mo.graph.graph import Node, Graph


class DivScalarFrontReplacer(FrontReplacementOp):
    op = '_div_scalar'
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        div_node = scalar_ops_replacer(graph, node, Div)
        return [div_node.id]


class GreaterScalarFrontReplacer(FrontReplacementOp):
    op = '_greater_scalar'
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        greater_node = scalar_ops_replacer(graph, node, Greater)
        return [greater_node.id]


class GreaterEqualScalarFrontReplacer(FrontReplacementOp):
    op = '_greater_equal_scalar'
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        greater_node = scalar_ops_replacer(graph, node, GreaterEqual)
        return [greater_node.id]


class EqualScalarFrontReplacer(FrontReplacementOp):
    op = '_equal_scalar'
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        equal_scalar_node = scalar_ops_replacer(graph, node, Equal)
        return [equal_scalar_node.id]


class NotEqualScalarFrontReplacer(FrontReplacementOp):
    op = '_not_equal_scalar'
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        not_equal_scalar_node = scalar_ops_replacer(graph, node, NotEqual)
        return [not_equal_scalar_node.id]


class LesserScalarFrontReplacer(FrontReplacementOp):
    op = '_lesser_scalar'
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        lesser_scalar_node = scalar_ops_replacer(graph, node, Less)
        return [lesser_scalar_node.id]


class LesserEqualScalarFrontReplacer(FrontReplacementOp):
    op = '_lesser_equal_scalar'
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        lesser_equal_scalar_node = scalar_ops_replacer(graph, node, LessEqual)
        return [lesser_equal_scalar_node.id]


class MinusScalarFrontReplacer(FrontReplacementOp):
    op = '_minus_scalar'
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        sub_node = scalar_ops_replacer(graph, node, Sub)
        return [sub_node.id]


class MulScalarFrontReplacer(FrontReplacementOp):
    op = '_mul_scalar'
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        mul_node = scalar_ops_replacer(graph, node, Mul)
        return [mul_node.id]


class PlusScalarFrontReplacer(FrontReplacementOp):
    op = '_plus_scalar'
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        add_node = scalar_ops_replacer(graph, node, Add)
        return [add_node.id]


class MinimumScalarFrontReplacer(FrontReplacementOp):
    op = '_minimum_scalar'
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        minimum_scalar_node = scalar_ops_replacer(graph, node, Minimum)
        return [minimum_scalar_node.id]


class MaximumScalarFrontReplacer(FrontReplacementOp):
    op = '_maximum_scalar'
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        maximum_scalar_node = scalar_ops_replacer(graph, node, Maximum)
        return [maximum_scalar_node.id]
