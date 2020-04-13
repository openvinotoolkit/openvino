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

from extensions.ops.elementwise import Mul, Add
from mo.front.common.replacement import FrontReplacementPattern
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, Node, rename_node


class Sub(FrontReplacementPattern):
    enabled = True
    graph_condition = [lambda graph: not graph.graph['cmd_params'].generate_experimental_IR_V10]

    @staticmethod
    def sub_to_add_replacement(sub: Node):
        # we execute this transformation for V10 IR later on middle phase despite graph_condition
        # so we prevent Sub replacement on shape-calculating sub-graphs
        if sub.in_port(0).data.get_value() is not None and sub.in_port(1).data.get_value() is not None:
            return

        graph = sub.graph
        name = sub.soft_get('name', sub.id)

        # keep Add name the same as Sub -- because of mathematical equality of output tensors
        rename_node(node=sub, name=name + '/to_be_removed')

        # reconnect Sub in(out)puts to Add
        add = Add(graph, {'name': name}).create_node()
        rename_node(add, name)

        sub.in_port(0).get_connection().set_destination(add.in_port(0))
        sub.in_port(1).get_connection().set_destination(add.in_port(1))
        sub.out_port(0).get_connection().set_source(add.out_port(0))

        # restore mathematical equivalence to Sub operation: Sub(A, B) = Add(A, Mul(B, -1))
        const_dtype = sub.soft_get('data_type', np.float32)
        negate = create_op_with_const_inputs(graph, Mul, {1: np.array(-1, dtype=const_dtype)}, {'name': name + '/neg_'})
        add.in_port(1).get_connection().insert_node(negate)

    def find_and_replace_pattern(self, graph: Graph):
        for sub in graph.get_op_nodes(op='Sub'):
            self.sub_to_add_replacement(sub)
