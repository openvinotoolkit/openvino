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
from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Node, Graph
from mo.ops.concat import Concat
from mo.ops.expand_dims import ExpandDims


class Pack(FrontReplacementOp):
    op = "Pack"
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        out_node = Concat(graph, {'axis': node.axis, 'in_ports_count': len(node.in_ports()),
                                  'name': node.name + '/Concat_', }).create_node()

        for ind in node.in_ports():
            expand_dims_node = ExpandDims(graph, {'expand_axis': int64_array([node.axis]),
                                                  'name': node.name + '/ExpandDims_'}).create_node()
            node.in_port(ind).get_connection().set_destination(expand_dims_node.in_port(0))
            expand_dims_node.out_port(0).connect(out_node.in_port(ind))
        # Replace edge from out port 0 of the matched node with a edge from node out_node.id with port 0.
        # The "explicit" version of the return value is: [(out_node.id, 0)])
        return [out_node.id]

