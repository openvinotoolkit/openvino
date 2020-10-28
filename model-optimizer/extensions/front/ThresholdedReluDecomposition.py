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

from extensions.ops.Cast import Cast
from extensions.ops.elementwise import Greater, Mul
from mo.front.common.partial_infer.utils import float_array
from mo.front.common.replacement import FrontReplacementPattern
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, rename_nodes
from mo.middle.passes.convert_data_type import data_type_str_to_np


class ThresholdedReluDecomposition(FrontReplacementPattern):
    """
    ThresholdedRelu(x, alpha) = x ? x > alpha : 0

    is replaced with

    ThresholdedRelu(x, alpha) = Mul(x, Cast(Greater(x, alpha), type=float))
    """
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(op='ThresholdedRelu'):
            name = node.soft_get('name', node.id)

            greater = create_op_with_const_inputs(graph, Greater, {1: float_array([node.alpha])})
            greater.in_port(0).connect(node.in_port(0).get_source())
            float_greater = Cast(graph,
                                 {'dst_type': data_type_str_to_np(graph.graph['cmd_params'].data_type)}).create_node()
            greater.out_port(0).connect(float_greater.in_port(0))

            mul = Mul(graph, {}).create_node()
            node.out_port(0).get_connection().set_source(mul.out_port(0))
            mul.in_port(0).connect(node.in_port(0).get_source())
            mul.in_port(1).connect(float_greater.out_port(0))

            rename_nodes([(node, name + '/TBR'), (mul, name)])
