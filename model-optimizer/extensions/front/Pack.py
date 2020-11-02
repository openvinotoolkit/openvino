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
from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementOp
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Node, Graph, rename_nodes
from mo.ops.concat import Concat
from mo.ops.unsqueeze import Unsqueeze


class Pack(FrontReplacementOp):
    op = "Pack"
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        out_node = Concat(graph, {'axis': node.axis, 'in_ports_count': len(node.in_ports())}).create_node()
        pack_name = node.soft_get('name', node.id)

        for ind in node.in_ports():
            unsqueeze_node = create_op_with_const_inputs(graph, Unsqueeze, {1: int64_array([node.axis])},
                                                         {'name': node.soft_get('name', node.id) + '/Unsqueeze'})
            node.in_port(ind).get_connection().set_destination(unsqueeze_node.in_port(0))
            unsqueeze_node.out_port(0).connect(out_node.in_port(ind))

        rename_nodes([(node, pack_name + '/TBR'), (out_node, pack_name)])
        return [out_node.id]

