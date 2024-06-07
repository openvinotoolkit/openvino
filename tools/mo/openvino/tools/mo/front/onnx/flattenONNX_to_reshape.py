# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.ReduceOps import ReduceProd
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph
from openvino.tools.mo.front.tf.graph_utils import create_op_node_with_second_input
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.const import Const
from openvino.tools.mo.ops.reshape import Reshape
from openvino.tools.mo.ops.shape import Shape
from openvino.tools.mo.utils.shape import node_to_get_shape_value_of_indices, new_shape_node_from_shape_nodes


class FlattenONNXToReshape(FrontReplacementSubgraph):
    """
    ONNX Flatten operation flattens the input tensor into a 2D matrix by given axis:

    Input of shape [d_0, d_1, ... d_n]
    Output of shape [d_0 X d_1 ... d_(axis-1),  d_axis X d_(axis+1) ... X dn]

    Corner case with axis=0: output shape will be [1, d_0 X d_1 ... X dn]
    """
    enabled = True

    def pattern(self):
        return dict(nodes=[('flatten', dict(op='FlattenONNX'))],
                    edges=[])

    def replace_sub_graph(self, graph: Graph, match: dict):
        node = match['flatten']
        name = node.soft_get('name', node.id)

        assert node.has_valid('axis'), 'Flatten {} should have `axis` attribute extracted, but it\'s not'.format(name)
        axis = node.axis

        reshape_node = Reshape(graph, {'name': node.id + '/Reshape'}).create_node()

        if axis == 0:
            dim = Const(graph, {'value': int64_array([1, -1]), 'name': reshape_node.name + '/shape'}).create_node()
        elif axis == 1:
            dim = Const(graph, {'value': int64_array([0, -1]), 'name': reshape_node.name + '/shape'}).create_node()
        else:
            shape = Shape(graph, {'name': name + '/input_shape'}).create_node()

            idxs = list(range(axis)) if axis > 0 else list(range(axis, 0))

            axis_shape_portion = node_to_get_shape_value_of_indices(shape, idxs)
            first_dims = create_op_node_with_second_input(graph, ReduceProd, int64_array([0]),
                                                          {'name': name + '/first_dims', 'keep_dims': True})
            second_dims = Const(graph, {'value': int64_array([-1]), 'name': name + '/second_dims'}).create_node()

            node.in_port(0).get_source().connect(shape.in_port(0))
            axis_shape_portion.out_port(0).connect(first_dims.in_port(0))

            order_of_dims = [first_dims, second_dims] if axis > 0 else [second_dims, first_dims]

            dim = new_shape_node_from_shape_nodes(order_of_dims)

        reshape_node.in_port(1).connect(dim.out_port(0))

        node.out_port(0).get_connection().set_source(reshape_node.out_port(0))
        node.in_port(0).get_connection().set_destination(reshape_node.in_port(0))
