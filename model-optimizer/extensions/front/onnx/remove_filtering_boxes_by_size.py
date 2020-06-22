"""
 Copyright (C) 2018-2020 Intel Corporation

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
from extensions.front.split_normalizer import AttributedVariadicSplitToVariadicSplit
from extensions.ops.range import Range
from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Graph
from mo.graph.graph import Node
from mo.ops.const import Const
from mo.ops.shape import Shape
from mo.ops.squeeze import Squeeze
from mo.utils.shape import node_to_get_batch_value


def skip_nodes_by_condition(current_node: Node, condition: callable):
    while condition(current_node):
        current_node = current_node.in_node()
    return current_node


class RemoveFilteringBoxesBySize(FrontReplacementSubgraph):
    """
    The transformation looks for a sub-graph that selects boxes with nonzero height and width. The output node of this
    sub-graph is a Cast node that produces indices of nodes to be preserved. The transformation creates a new sub-graph
    that produces a tensor with values from 0 to input.shape[0] to select all boxes. The output of this sub-graph will
    be used in the NonMaxSuppression so the implementation of this layer should ignore boxes with negative sizes.
    """
    enabled = True
    force_clean_up = True

    def run_after(self):
        return [AttributedVariadicSplitToVariadicSplit]

    def pattern(self):
        return dict(
            nodes=[
                ('split', dict(op='VariadicSplit')),
                ('sub_1', dict(op='Sub')),
                ('sub_2', dict(op='Sub')),
                ('add_1', dict(op='Add')),
                ('add_2', dict(op='Add')),
                ('concat', dict(op='Concat')),
                ('split_2', dict(op='VariadicSplit')),
                ('squeeze_1', dict(op='Squeeze')),
                ('squeeze_2', dict(op='Squeeze')),
                ('less_1', dict(op='Less')),
                ('less_2', dict(op='Less')),
                ('not_1', dict(op='LogicalNot')),
                ('not_2', dict(op='LogicalNot')),
                ('cast_11', dict(op='Cast')),
                ('cast_12', dict(op='Cast')),
                ('cast_21', dict(op='Cast')),
                ('cast_22', dict(op='Cast')),
                ('and', dict(op='LogicalAnd')),
                ('cast_31', dict(op='Cast')),
                ('cast_32', dict(op='Cast')),
                ('nonzero', dict(op='NonZero')),
                ('transpose', dict(op='Transpose')),
                ('squeeze', dict(op='Squeeze')),
                ('cast', dict(op='Cast')),
            ],
            edges=[
                ('split', 'sub_1', {'in': 0, 'out': 2}),
                ('split', 'sub_1', {'in': 1, 'out': 0}),
                ('split', 'sub_2', {'in': 0, 'out': 3}),
                ('split', 'sub_2', {'in': 1, 'out': 1}),
                ('sub_1', 'add_1', {}),
                ('sub_2', 'add_2', {}),
                ('split', 'concat', {'in': 0, 'out': 0}),
                ('split', 'concat', {'in': 1, 'out': 1}),
                ('add_1', 'concat', {'in': 2, 'out': 0}),
                ('add_2', 'concat', {'in': 3, 'out': 0}),
                ('concat', 'split_2', {}),
                ('split_2', 'squeeze_1', {'in': 0, 'out': 2}),
                ('split_2', 'squeeze_2', {'in': 0, 'out': 3}),
                ('squeeze_1', 'less_1', {}),
                ('squeeze_2', 'less_2', {}),
                ('less_1', 'not_1', {}),
                ('less_2', 'not_2', {}),
                ('not_1', 'cast_11', {}),
                ('cast_11', 'cast_12', {}),
                ('not_2', 'cast_21', {}),
                ('cast_21', 'cast_22', {}),
                ('cast_12', 'and', {}),
                ('cast_22', 'and', {}),
                ('and', 'cast_31', {}),
                ('cast_31', 'cast_32', {}),
                ('cast_32', 'nonzero', {}),
                ('nonzero', 'transpose', {}),
                ('transpose', 'squeeze', {}),
                ('squeeze', 'cast', {}),
            ])

    def replace_sub_graph(self, graph: Graph, match: dict):
        source_connection = match['split'].in_port(0).get_connection()
        source_node = source_connection.get_source().node
        cast_node = match['cast']

        range_node = Range(graph, {'name': source_node.id + '/Range'}).create_node()
        start_node = Const(graph, {'name': range_node.id + '/Start', 'value': int64_array(0)}).create_node()

        step_node = Const(graph, {'name': range_node.id + '/Step', 'value': int64_array(1)}).create_node()
        input_shape_node = Shape(graph, {'name': start_node.id + '/Shape'}).create_node()
        input_shape_node.in_port(0).connect(source_node.out_port(0))

        limit_node_1D = node_to_get_batch_value(input_shape_node)
        limit_node = create_op_node_with_second_input(graph, Squeeze, int64_array([0]),
                                                      {'name': source_node.id + '/batch_0D_value'}, limit_node_1D)

        range_node.in_port(0).connect(start_node.out_port(0))
        range_node.in_port(1).connect(limit_node.out_port(0))
        range_node.in_port(2).connect(step_node.out_port(0))
        cast_node.out_port(0).get_connection().set_source(range_node.out_port(0))

        graph.remove_nodes_from([node.id for node in match.values()])
