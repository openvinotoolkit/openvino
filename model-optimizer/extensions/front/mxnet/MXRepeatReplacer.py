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

from extensions.front.rank_decomposer import RankDecomposer
from extensions.ops.elementwise import Add, Sub, Mul
from extensions.ops.gather import Gather
from extensions.ops.rank import Rank
from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementPattern
from mo.front.tf.graph_utils import create_op_node_with_second_input, create_op_with_const_inputs
from mo.graph.graph import Graph, Node, rename_node
from mo.ops.broadcast import Broadcast
from mo.ops.const import Const
from mo.ops.reshape import Reshape
from mo.ops.shape import Shape
from mo.ops.tile import Tile
from mo.ops.unsqueeze import Unsqueeze
from mo.utils.shape import get_canonical_axis_index_node, new_shape_node_from_shape_nodes, \
    get_shape_values_by_range_idxs


class MXRepeatReplacer(FrontReplacementPattern):
    """
        The transformation converts MXRepeat operation to Unsqueeze -> Tile -> Reshape.
    """

    enabled = True
    force_clean_up = True

    def run_before(self):
        return [RankDecomposer]

    def find_and_replace_pattern(self, graph: Graph):
        for mxrepeat in graph.get_op_nodes(op='MXRepeat'):
            self.mxrepeat_decomposition(mxrepeat)

    @staticmethod
    def mxrepeat_decomposition(node: Node):
        graph = node.graph
        name = node.soft_get('name', node.id)

        rename_node(node, name + '/to_be_removed')

        # Unqueeze
        input_rank = Rank(graph, {'name': name + '/Rank'}).create_node()
        node.in_port(0).get_source().connect(input_rank.in_port(0))

        axis = get_canonical_axis_index_node(input_rank, node.axis)
        unsqueeze_axis = create_op_node_with_second_input(
            graph, Add, int64_array([1]), {'name': name + '/Unsqueeze/Axis'}, input_node=axis)

        unsqueeze = Unsqueeze(graph, {'name': name + '/Unsqueeze'}).create_node()
        unsqueeze.in_port(1).connect(unsqueeze_axis.out_port(0))

        # Tile (1, 1, ..., repeats, ..., 1)
        # we generate tile array according to the following table:

        # parts:       |      first      |  repeats |  second     |
        # i:           | 0, 1, ..., axis,| axis + 1,| ..., rank+1 |
        # tile_array:  | 1, 1, ...,  1  ,| repeats ,| ...,   1    |

        one = Const(graph, {'name': name + '/Broadcast/One', 'value': int64_array([1])}).create_node()
        first_ones = Broadcast(graph, {'name': name + '/Broadcast/Ones_first_part'}).create_node()
        first_ones.in_port(0).connect(one.out_port(0))
        first_ones.in_port(1).connect(unsqueeze_axis.out_port(0))

        repeats = Const(graph, {'name': name + '/repeats', 'value': int64_array([node.repeats])}).create_node()

        second_ones = Broadcast(graph, {'name': name + '/Broadcast/Ones_second_part'}).create_node()
        second_part_broadcast_shape = Sub(graph, {'name': name + '/Broadcast/Shape/second_part'}).create_node()
        second_part_broadcast_shape.in_port(0).connect(input_rank.out_port(0))
        second_part_broadcast_shape.in_port(1).connect(unsqueeze_axis.out_port(0))
        second_ones.in_port(0).connect(one.out_port(0))
        second_ones.in_port(1).connect(second_part_broadcast_shape.out_port(0))

        tile_repeats = new_shape_node_from_shape_nodes([first_ones, repeats, second_ones])
        tile = Tile(graph, {'name': name + '/Tile'}).create_node()
        tile.in_port(1).connect(tile_repeats.out_port(0))

        # Reshape (input_shape[:axis], input_shape[axis] * repeats, input_shape[axis+1:])
        # we generate reshape dim array according to the following table:

        # parts:       |    first   |                rep           |  second   |
        # i:           | 0, 1, ... ,|               axis,          | ..., rank |
        # dim_array:   | inp_sh[i] ,| input_shape[axis] * repeats ,| inp_sh[i] |

        input_shape = Shape(graph, {'name': name + '/Shape'}).create_node()
        node.in_port(0).get_source().connect(input_shape.in_port(0))

        first_input_shape_part = get_shape_values_by_range_idxs(
            input_shape, input_rank, begin=0, end=node.axis, include_begin=True, include_end=False)

        original_axis_dim = create_op_with_const_inputs(
            graph, Gather, {2: int64_array(0)}, {'name': name + '/OriginalDim'}, input_node=input_shape)
        original_axis_dim.in_port(1).connect(axis.out_port(0))

        repeated_dimention = Mul(graph, {'name': name + '/RepeatedDim'}).create_node()
        repeated_dimention.in_port(0).connect(original_axis_dim.out_port(0))
        repeated_dimention.in_port(1).connect(repeats.out_port(0))

        second_input_shape_part = get_shape_values_by_range_idxs(
            input_shape, input_rank, begin=node.axis, end=-1, include_begin=False, include_end=True)

        output_shape = new_shape_node_from_shape_nodes(
            [first_input_shape_part, repeated_dimention, second_input_shape_part])

        reshape = Reshape(graph, {'name': name}).create_node()
        rename_node(reshape, name)
        reshape.in_port(1).connect(output_shape.out_port(0))

        # Final connections
        node.in_port(0).get_connection().set_destination(unsqueeze.in_port(0))
        tile.in_port(0).connect(unsqueeze.out_port(0))
        reshape.in_port(0).connect(tile.out_port(0))
        node.out_port(0).get_connection().set_source(reshape.out_port(0))
