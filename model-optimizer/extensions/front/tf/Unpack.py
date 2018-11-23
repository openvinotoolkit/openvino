"""
 Copyright (c) 2018 Intel Corporation

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

import networkx as nx

from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Node, insert_node_after
from mo.ops.squeeze import Squeeze


class Unpack(FrontReplacementOp):
    """
    The Unpack from TF operation removes dimension over which the unpack is performed. The "Split" layer of IE doesn't
    do that. This replacer adds squeeze operation for each output of the Unpack nodes to remove the dimension.
    """
    op = "Unpack"
    enabled = True

    def nodes_to_remove(self, graph: nx.MultiDiGraph, match: dict):
        # do not remove matched node
        return []

    def replace_op(self, graph: nx.MultiDiGraph, node: Node):
        for ind in range(len(node.out_nodes())):
            squeeze_node = Squeeze(graph, dict(squeeze_dims=[node.axis], name=node.name + '/Squeeze_')).create_node([])
            insert_node_after(node, squeeze_node, ind)

        # do not replace any output edge
        return []
