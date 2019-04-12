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
from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Node, Graph
from mo.ops.concat import Concat
from mo.ops.const import Const
from mo.ops.expand_dims import ExpandDims


class Pack(FrontReplacementOp):
    op = "Pack"
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        expand_dims_nodes = list()
        expand_axis_node = Const(graph, dict(value=node.axis)).create_node([])
        for ind, edge_attrs in node.in_edges().items():
            expand_dims_nodes.append(ExpandDims(graph, dict(name=node.name + '/ExpandDims_')).
                                     create_node([(node.in_node(ind), edge_attrs['out']), expand_axis_node]))

        out_node = Concat(graph, dict(name=node.name + '/Concat_',
                                      axis=node.axis,
                                      in_ports_count=len(expand_dims_nodes))).create_node(expand_dims_nodes)
        # Replace edge from out port 0 of the matched node with a edge from node out_node.id with port 0.
        # The "explicit" version of the return value is: [(out_node.id, 0)])
        return [out_node.id]
