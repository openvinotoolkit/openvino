# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.ops.elementwise import Mul
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.concat import Concat
from openvino.tools.mo.ops.const import Const
from openvino.tools.mo.ops.reshape import Reshape
from openvino.tools.mo.ops.shape import Shape
from openvino.tools.mo.utils.shape import get_shape_values_by_indices_node, get_shape_values_by_range_idxs


class MXReshapeToReshape(FrontReplacementSubgraph):
    """
    Generate subgraph which is equivalent for transform of -2 -3 or -4 options in reshape dims attribute.
    -2 copy all/remainder of the input dimensions to the output shape.
        Example: input shape = (2,3,4), shape = (2,-2,1,1), output shape = (2,3,4,1,1)
    -3 use the product of two consecutive dimensions of the input shape as the output dimension.
        Example: input shape = (2,3,4), shape = (-3,4), output shape = (6,4)
    -4 split one dimension of the input into two dimensions passed subsequent to -4 in shape (can contain -1).
        Example: input shape = (2,3,4), shape = (-4,1,2,-2), output shape =(1,2,3,4)
    """
    enabled = True

    def pattern(self):
        return dict(
            nodes=[
                ('mxreshape', dict(op='MXReshape', dim=lambda node_dim: node_dim is not None and len(set(node_dim).intersection({-2, -3, -4})) != 0, reverse=False))
            ],
            edges=[]
        )

    def resolve_handlers(self, case):
        return {
            -2: self.resolve_minus2,
            -3: self.resolve_minus3,
            -4: self.resolve_minus4,
        }.get(case, self.resolve_const_shape)

    def resolve_minus2(self, shape_node, input_index, reshape_index, dims):
        rank_node = Shape(shape_node.graph, dict(name=shape_node.id + '/RankShapeMXReshapeMinus2')).create_node()
        rank_node.in_port(0).connect(shape_node.out_port(0))
        shape_values_node = get_shape_values_by_range_idxs(shape=shape_node, rank=rank_node,
                                                           begin=input_index, end=-1,
                                                           include_begin=True, include_end=True)
        input_index = None
        reshape_index = reshape_index + 1
        return input_index, reshape_index, dims, shape_values_node

    def resolve_minus3(self, shape_node, input_index, reshape_index, dims):
        shape_indexes_node1 = Const(shape_node.graph, dict(name=shape_node.id + '/ShapeMinus3_index_const1_' + str(input_index),
                                                           value=int64_array([input_index]))).create_node()
        dims_node1 = get_shape_values_by_indices_node(shape_node, shape_indexes_node1)

        shape_indexes_node2 = Const(shape_node.graph, dict(name=shape_node.id + '/ShapeMinus3_index_const2_' + str(input_index),
                                                           value=int64_array([input_index + 1]))).create_node()
        dims_node2 = get_shape_values_by_indices_node(shape_node, shape_indexes_node2)

        mul_node = Mul(shape_node.graph, dict(name=shape_node.id + '/MulMinus3_' + str(input_index))).create_node()

        mul_node.in_port(0).connect(dims_node1.out_port(0))
        mul_node.in_port(1).connect(dims_node2.out_port(0))

        input_index = input_index + 2
        reshape_index = reshape_index + 1
        return input_index, reshape_index, dims, mul_node

    def resolve_minus4(self, shape_node, input_index, reshape_index, dims):
        shape_const_node = Const(shape_node.graph, dict(name=shape_node.id + '/ShapeMinus4_index_const_' + str(input_index),
                                                        value=np.take(dims, [reshape_index + 1, reshape_index + 2]))).create_node()
        input_index = input_index + 2
        reshape_index = reshape_index + 3
        return input_index, reshape_index, dims, shape_const_node

    def resolve_const_shape(self, shape_node, input_index, reshape_index, dims):
        dim_const_node = Const(shape_node.graph, dict(name=shape_node.id + '/DimConst_' + str(reshape_index),
                                                      value=[dims[reshape_index]])).create_node()
        input_index = input_index + 1 if input_index != None else None
        reshape_index = reshape_index + 1
        return input_index, reshape_index, dims, dim_const_node

    def resolve(self, input_index, reshape_index, dims, input_shape_node, output_dims_nodes):

        resolve_handler = self.resolve_handlers(dims[reshape_index])
        input_index, reshape_index, dims, dims_node = resolve_handler(input_shape_node, input_index,
                                                                      reshape_index, dims)
        output_dims_nodes.append(dims_node)
        return input_index, reshape_index, output_dims_nodes

    def replace_sub_graph(self, graph: Graph, match: dict):
        node = match['mxreshape']

        input_index = 0
        reshape_index = 0
        shape_node = Shape(graph, dict(name=node.id + '/ShapeMXReshape')).create_node()
        shape_node.in_port(0).connect(node.in_port(0).get_source())
        output_dims_nodes = []
        for d in node.dim:
            if reshape_index < len(node.dim):
                input_index, reshape_index, output_dims_nodes = self.resolve(input_index, reshape_index, node.dim, shape_node, output_dims_nodes)

        concat_node = Concat(shape_node.graph, dict(name=shape_node.id + '/ConcatMXReshape_', axis=0,
                                                    in_ports_count=len(output_dims_nodes))).create_node()

        for in_port_index, dim_node in enumerate(output_dims_nodes):
            concat_node.in_port(in_port_index).connect(dim_node.out_port(0))

        reshape_node = Reshape(graph, dict(name=node.id + '/Reshape_')).create_node()
        reshape_node.in_port(1).connect(concat_node.out_port(0))
        node.in_port(0).get_connection().set_destination(reshape_node.in_port(0))
        node.out_port(0).get_connection().set_source(reshape_node.out_port(0))
