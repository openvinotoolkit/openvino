# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from extensions.front.onnx.quantize_dequantize_linear import QuantizeDequantizeLinear
from extensions.ops.Cast import Cast
from extensions.ops.elementwise import Mul, Sub
from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Graph, Node, rename_nodes
from mo.middle.passes.convert_data_type import data_type_str_to_np
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.front.common.partial_infer.utils import float_array, int64_array

from mo.ops.shape import Shape
from extensions.ops.rank import Rank
from extensions.ops.range import Range
from extensions.ops.gather import Gather
from extensions.ops.elementwise import GreaterEqual
from extensions.ops.scatter import ScatterElementsUpdate
from mo.ops.reshape import Reshape


class DequantizeLinearResolver(FrontReplacementOp):
    """
    DequantizeLinear can be replace with the following formula: y = (x - x_zero_point) * x_scale
    """
    op = "DequantizeLinear"
    enabled = True

    def run_after(self):
        return [QuantizeDequantizeLinear]

    def replace_op(self, graph: Graph, node: Node):
        node_name = node.soft_get('name', node.id)
        model_data_type = data_type_str_to_np(graph.graph['cmd_params'].data_type)
        cast = Cast(graph, {'dst_type': model_data_type, 'name': node_name + '/Cast'}).create_node()

        axis = node.soft_get('axis', node.id)
        if axis is not None and axis != 1:
            data_shape_node = Shape(graph, {'name': node_name + '/Shape'}).create_node()
            node.in_port(0).get_source().connect(data_shape_node.in_port(0))
            gather_node = create_op_with_const_inputs(graph, Gather, {1: int64_array([axis])},
                                                      {'name': node_name + '/Gather'})
            data_shape_node.out_port(0).connect(gather_node.in_port(0))

            rank_node = Rank(graph, {'name': node_name + '/Rank'}).create_node()
            node.in_port(0).get_source().connect(rank_node.in_port(0))

            range_node = create_op_with_const_inputs(graph, Range, {0: int64_array([1]),
                                                                    2: int64_array([1])},
                                                     {'name': node_name + '/Range'})
            rank_node.out_port(0).connect(range_node.in_port(1))
            greater_equal_node = create_op_with_const_inputs(graph, GreaterEqual, {1: int64_array([1])},
                                                             {'name': node_name + '/GreaterEqual'})
            range_node.out_port(0).connect(greater_equal_node.in_port(0))

            scatter_elements_node = create_op_with_const_inputs(graph, ScatterElementsUpdate, {1: int64_array([axis])},
                                                                {'name': node_name + '/ScatterElements'})

            cast_node = Cast(graph, {'dst_type': np.int64, 'name': node_name + '/Cast'}).create_node()
            greater_equal_node.out_port(0).connect(cast_node.in_port(0))
            cast_node.out_port(0).connect(scatter_elements_node.in_port(0))

            gather_node.out_port(0).connect(scatter_elements_node.in_port(2))
            reshape_node = Reshape(graph, {'name': node_name + '/Reshape_low'}).create_node()
            scatter_elements_node.out_port(0).connect(reshape_node.in_port(1))

            reshape_node.out_port(0).connect(cast.in_port(0))
            node.in_port(0).get_connection().set_destination(reshape_node.in_port(0))
        else:
            node.in_port(0).get_connection().set_destination(cast.in_port(0))
        mul = Mul(graph, {}).create_node()

        if node.is_in_port_connected(2):
            sub = Sub(graph, {'name': node_name + '/Sub'}).create_node()
            cast.out_port(0).connect(sub.in_port(0))
            node.in_port(2).get_connection().set_destination(sub.in_port(1))
            sub.out_port(0).connect(mul.in_port(0))
        else:
            cast.out_port(0).connect(mul.in_port(0))

        node.in_port(1).get_connection().set_destination(mul.in_port(1))
        rename_nodes([(node, node_name + '/TBD'), (mul, node_name)])

        return [mul.id]
