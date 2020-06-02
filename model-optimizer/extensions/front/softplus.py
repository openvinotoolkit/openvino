"""
 Copyright (C) 2020 Intel Corporation

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

from extensions.ops.activation_ops import Exp, Log
from extensions.ops.elementwise import Add
from mo.front.common.partial_infer.utils import float_array
from mo.front.common.replacement import FrontReplacementOp
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Graph, rename_nodes


class SoftPlus(FrontReplacementOp):
    """
    The transformation replaces SoftPlus(x) with log(1.0 + exp(x)).
    """
    op = 'SoftPlus'
    enabled = True

    def replace_sub_graph(self, graph: Graph, match: dict):
        softplus = match['op']

        name = softplus.soft_get('name', softplus.id)
        exp_node = Exp(graph, {'name': name + '/Exp'}).create_node()
        add_node = create_op_node_with_second_input(graph, Add, float_array([1.0]), {'name': name + '/Add'})
        log_node = Log(graph, {'name': name + '/Log'}).create_node()
        rename_nodes([(softplus, name + '/Log'), (log_node, name)])

        softplus.in_port(0).get_connection().set_destination(exp_node.in_port(0))
        add_node.in_port(0).connect(exp_node.out_port(0))
        log_node.in_port(0).connect(add_node.out_port(0))
        softplus.out_port(0).get_connection().set_source(log_node.out_port(0))
