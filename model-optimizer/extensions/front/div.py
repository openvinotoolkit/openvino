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

from extensions.ops.elementwise import Mul, Pow
from mo.front.common.replacement import FrontReplacementPattern
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, Node, rename_node


class Div(FrontReplacementPattern):
    # This transformation is called directly from the 'model-optimizer/extensions/middle/fusings.py' transformation
    enabled = False

    @staticmethod
    def div_to_mul_replacement(div: Node):
        # we execute this transformation for V10 IR later on middle phase despite graph_condition
        # so we prevent Div replacement on shape-calculating sub-graphs
        if div.in_port(0).data.get_value() is not None and div.in_port(1).data.get_value() is not None:
            return

        # cannot replace Div with Mul when the divisor is integer because the reciprocal number will be 0
        value = div.in_port(1).data.get_value()
        if value is not None and type(value.item(0)) == int:
            return

        graph = div.graph
        name = div.soft_get('name', div.id)

        # keep Mul name the same as Div -- because of mathematical equality of output tensors
        rename_node(node=div, name=name + '/to_be_removed')

        # reconnect Div in(out)puts to Mul
        mul = Mul(graph, {'name': name}).create_node()
        rename_node(mul, name)

        div.in_port(0).get_connection().set_destination(mul.in_port(0))
        div.in_port(1).get_connection().set_destination(mul.in_port(1))
        div.out_port(0).get_connection().set_source(mul.out_port(0))

        # restore mathematical equivalence to Div operation: Div(A, B) = Mul(A, Pow(B, -1))
        reciprocal = create_op_with_const_inputs(graph, Pow, {1: np.float64(-1)}, {'name': name + '/reciprocal_'})
        mul.in_port(1).get_connection().insert_node(reciprocal)

    def find_and_replace_pattern(self, graph: Graph):
        for div in graph.get_op_nodes(op='Div'):
            self.div_to_mul_replacement(div)
