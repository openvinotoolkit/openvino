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
from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementOp
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Graph
from mo.ops.reshape import Reshape


class ArgMaxFlatten(FrontReplacementOp):
    """
    The ArgMax layer in Caffe may have non-specified 'axis' attribute. In this case it should flatten input data before
    calculating ArgMax.
    """
    op = "ArgMax"
    enabled = True

    def replace_sub_graph(self, graph: Graph, match: dict):
        argmax_node = match['op']
        if not argmax_node.has_valid('axis'):
            flatten_node = create_op_node_with_second_input(graph, Reshape, int64_array([0, 1, -1]),
                                                            dict(name=argmax_node.name + '/Flatten'))
            argmax_node.in_port(0).get_connection().insert_node(flatten_node)
            argmax_node.axis = 2
