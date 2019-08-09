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


import numpy as np

from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Node, Graph
from mo.ops.const import Const
from mo.ops.squeeze import Squeeze


class Unpack(FrontReplacementOp):
    """
    The Unpack from TF operation removes dimension over which the unpack is performed. The "Split" layer of IE doesn't
    do that. This replacer adds squeeze operation for each output of the Unpack nodes to remove the dimension.
    """
    op = "Unpack"
    enabled = True

    def nodes_to_remove(self, graph: Graph, match: dict):
        # do not remove matched node
        return []

    def replace_op(self, graph: Graph, node: Node):
        for out_port in node.out_ports().values():
            squeeze_node = Squeeze(graph, dict(name=node.name + '/Squeeze_')).create_node([])
            dims_node = Const(graph, {'value': np.array(node.axis), 'name': node.name + '/Squeeze_axis'}).create_node()

            out_port.get_connection().insert_node(squeeze_node)
            dims_node.out_port(0).connect(squeeze_node.in_port(1))
        # do not replace any output edge
        return []
