# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.rank_decomposer import RankDecomposer
from openvino.tools.mo.ops.ReduceOps import ReduceProd
from openvino.tools.mo.ops.rank import Rank
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph
from openvino.tools.mo.front.tf.graph_utils import create_op_node_with_second_input
from openvino.tools.mo.graph.graph import Graph, rename_nodes
from openvino.tools.mo.ops.const import Const
from openvino.tools.mo.ops.reshape import Reshape
from openvino.tools.mo.ops.shape import Shape
from openvino.tools.mo.utils.shape import new_shape_node_from_shape_nodes, get_shape_values_by_range_idxs


class FlattenToReshape(FrontReplacementSubgraph):
    """
    Flatten operation flattens the input tensor according to given `axis` and `end_axis` parameters:

    Input of shape [d_0, d_1, ... d_n]
    Output of shape [d_0, d_1, ... , d_(axis-1), d_axis X ... X d_(end_axis), d_(end_axis + 1), ... , dn]
    """
    enabled = True

    def run_before(self):
        return [RankDecomposer]

    def pattern(self):
        return dict(nodes=[
            ('flatten', dict(op='Flatten'))
        ],
            edges=[])

    def replace_sub_graph(self, graph: Graph, match: dict):
        node = match['flatten']
        name = node.soft_get('name', node.id)

        assert node.has_valid('axis'), 'Flatten {} has no mandatory `axis` attribute'.format(name)
        assert node.has_valid('end_axis'), 'Flatten {} has no mandatory `end_axis` attribute'.format(name)

        axis = node.axis
        end_axis = node.end_axis

        if end_axis == -1 and axis >= 0:
            begin_dims = Const(graph, {'value': int64_array([0] * axis)}).create_node()
            middle_dim = Const(graph, {'value': int64_array([-1])}).create_node()
            end_dims = Const(graph, {'value': int64_array([])}).create_node()
        else:
            rank = Rank(graph, {'name': name + '/input_rank'}).create_node()
            node.in_port(0).get_source().connect(rank.in_port(0))

            shape = Shape(graph, {'name': name + '/input_shape'}).create_node()
            node.in_port(0).get_source().connect(shape.in_port(0))

            begin_dims = get_shape_values_by_range_idxs(
                shape=shape, rank=rank, begin=0, end=axis)
            middle_dims = get_shape_values_by_range_idxs(
                shape=shape, rank=rank, begin=axis, end=end_axis, include_end=True)
            end_dims = get_shape_values_by_range_idxs(
                shape=shape, rank=rank, begin=end_axis, end=-1, include_begin=False, include_end=True)

            middle_dim = create_op_node_with_second_input(graph, ReduceProd, int64_array([0]), {'keep_dims': True})
            middle_dims.out_port(0).connect(middle_dim.in_port(0))

        dim = new_shape_node_from_shape_nodes([begin_dims, middle_dim, end_dims])

        original_name = node.soft_get('name')
        abandoned_name = original_name + '/ShouldBeDeleted'
        reshape_node = Reshape(graph, {}).create_node()
        # Keep node with the same name to avoid confuse with renaming
        rename_nodes([(node, abandoned_name), (reshape_node, original_name)])
        reshape_node.in_port(1).connect(dim.out_port(0))

        node.out_port(0).get_connection().set_source(reshape_node.out_port(0))
        node.in_port(0).get_connection().set_destination(reshape_node.in_port(0))
