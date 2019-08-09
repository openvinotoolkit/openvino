"""
 Copyright (c) 2019 Intel Corporation

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
from extensions.ops.Log import LogOp
from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Graph, Node
from mo.ops.softmax import Softmax


class LogSoftmaxFrontReplacer(FrontReplacementOp):
    """
    Replace LogSoftmax operation by Softmax -> Log.
    """
    op = "LogSoftmax"
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        axis = -1
        if 'axis' in node.pb.attr:
            axis = node.pb.attr['axis'].i

        log = LogOp(graph, {'name': node.name + '/Log_'}).create_node()
        softmax = Softmax(graph, {'axis': axis, 'name': node.name + '/SoftMax_'}).create_node()

        # Connect nodes: input -> Softmax -> Log
        node.in_port(0).get_connection().set_destination(softmax.in_port(0))
        log.in_port(0).get_connection().set_source(softmax.out_port(0))

        # The "explicit" version of the return value is: [(out_node.id, 0)])
        return [log.id]
