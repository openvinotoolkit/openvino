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
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.reshape import Reshape


class FlattenONNXToReshape(MiddleReplacementPattern):
    """
    Replaces FlattenONNX to Reshape
    """

    enabled = True

    def pattern(self):
        return dict(
            nodes=[
                ('op', dict(op='FlattenONNX')),
            ],
            edges=[])

    def replace_pattern(self, graph: Graph, match: dict):
        node = match['op']

        # TODO FIXME implement reshape-able way to determine operation output shape
        reshape_node = create_op_node_with_second_input(graph, Reshape, node.out_port(0).data.get_shape(),
                                                        {'name': node.id + '/Dims'})

        node.in_port(0).get_connection().set_destination(reshape_node.in_port(0))
        node.out_port(0).get_connection().set_source(reshape_node.out_port(0))
