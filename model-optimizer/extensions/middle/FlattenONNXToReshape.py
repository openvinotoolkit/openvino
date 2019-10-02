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
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.reshape import Reshape


class FlattenONNXToReshape(MiddleReplacementPattern):
    """
    Replaces FlattenONNX to Reshape
    """

    enabled = True
    force_clean_up = True

    def pattern(self):
        return dict(
            nodes=[
                ('op', dict(op='FlattenONNX')),
            ],
            edges=[])

    def replace_pattern(self, graph: Graph, match: dict):
        node = match['op']
        assert node.has_valid('axis')
        axis = node.axis

        if axis == 0:
            dim = int64_array([1, -1])
        elif axis == 1:
            dim = int64_array([0, -1])
        else:
            # FIXME: can not simply express that kind of flattening with special Reshape symbols,
            # TODO: create sub-graph that calculates `dim`
            dim = node.out_port(0).data.get_shape()

        reshape_node = create_op_node_with_second_input(graph, Reshape, dim, {'name': node.id + '/Dims'})
        node.out_port(0).get_connection().set_source(reshape_node.out_port(0))
        node.in_port(0).get_connection().set_destination(reshape_node.in_port(0))
