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
from extensions.ops.elementwise import Add
from mo.ops.scale_shift import ScaleShiftOp


class AxpyToSSandAdd(FrontReplacementOp):
    """
    Replaces Axpy layer with ScaleShift and Add.
    """
    op = "Axpy"
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        in_node_0 = node.in_node(0)
        in_node_1 = node.in_node(1)
        in_node_2 = node.in_node(2)

        ss = ScaleShiftOp(graph, {'name': node.id + "/ScaleShift_", 'axis': 0})
        scale_shift = ss.create_node(inputs=[in_node_1, in_node_0])

        el = Add(graph, {'name': node.id + "/Add_"})
        el_node = el.create_node(inputs=[scale_shift, in_node_2])

        return [el_node.id]
