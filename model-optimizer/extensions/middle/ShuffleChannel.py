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

from extensions.ops.transpose import Transpose
from mo.front.common.partial_infer.utils import int64_array
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.reshape import Reshape
from mo.utils.error import Error


class ShuffleChannel(MiddleReplacementPattern):
    """
    Replaces Caffe ShuffleChannel with Reshapes and Transpose layers
    """

    enabled = True

    def run_before(self):
        from extensions.middle.pass_separator import MiddleFinish
        return [MiddleFinish]

    def pattern(self):
        return dict(
            nodes=[
                ('op', dict(op='ShuffleChannel')),
            ],
            edges=[
            ])

    def replace_pattern(self, graph: Graph, match: dict):
        if graph.graph['layout'] != "NCHW":
            return

        node = match['op']

        in_node = node.in_node(0)
        out_node = node.out_node(0)
        group = int(node['group'])

        rows = group
        cols = in_node.shape[1] // group

        if rows * cols != in_node.shape[1]:
            raise Error("Group {} should divide input channels number {} without reminder for node {}"
                        "".format(group, in_node.shape[1], node.id))

        reshape_split_node = create_op_node_with_second_input(graph, Reshape,
                                                              int64_array([in_node.shape[0], rows, cols, -1]),
                                                              {'name': node.id + '/Reshape_split_'})

        transpose_node = create_op_node_with_second_input(graph, Transpose, int64_array([0, 2, 1, 3]),
                                                          {'name': node.id + '/Transpose_'}, reshape_split_node)

        reshape_concat = create_op_node_with_second_input(graph, Reshape, out_node.shape,
                                                          {'name': node.id + '/Reshape_concat_'}, transpose_node)

        node.in_port(0).get_connection().set_destination(reshape_split_node.in_port(0))
        node.out_port(0).get_connection().set_source(reshape_concat.out_port(0))
