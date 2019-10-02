"""
 Copyright (c) 2019 Intel Corporation

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

from extensions.ops.elementwise import Div, Greater, GreaterEqual, Equal, NotEqual, Sub, Mul, Add, Less, LessEqual, Minimum, Maximum
from mo.front.common.replacement import FrontReplacementOp
from mo.front.mxnet.extractors.utils import scalar_ops_replacer
from mo.graph.graph import Node, Graph


class DivScalarFrontReplacer(FrontReplacementOp):
    op = '_div_scalar'
    enabled = True

    def run_before(self):
        from extensions.front.div import Div
        return [Div]

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

    def run_before(self):
        from extensions.front.sub import Sub
        return [Sub]

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
