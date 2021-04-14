# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from extensions.front.onnx.quantize_dequantize_linear import QuantizeDequantizeLinear
from extensions.ops.Cast import Cast
from extensions.ops.elementwise import Mul
from extensions.ops.fakequantize import FakeQuantize
from mo.front.common.partial_infer.utils import float_array, int64_array
from mo.front.common.replacement import FrontReplacementOp
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, Node, rename_nodes
from mo.ops.const import Const
from mo.utils.error import Error

from mo.ops.shape import Shape
from extensions.ops.rank import Rank
from extensions.ops.range import Range
from extensions.ops.gather import Gather
from extensions.ops.elementwise import GreaterEqual
from extensions.ops.scatter import ScatterElementsUpdate
from mo.ops.reshape import Reshape


class QuantizeLinearResolver(FrontReplacementOp):
    """
    Replaces QuantizeLinear with FakeQuantize
    """
    op = "QuantizeLinear"
    enabled = True

    def run_after(self):
        return [QuantizeDequantizeLinear]

    def replace_op(self, graph: Graph, node: Node):
        node_name = node.soft_get('name', node.id)

        if node.is_in_port_connected(2):
            zerop = node.in_port(2).get_source().node
        else:
            zerop = Const(graph, {'value': np.array(0, dtype=np.uint8), 'name': node_name + '/ZeroPoint'}).create_node()

        assert zerop.soft_get('type') == 'Const', 'only constant for zero_point is supported for QuantizeLinear'
        zero_point_type = zerop.value.dtype
        # data type affects range of output values: [-128..127] or [0..255]
        if zero_point_type == np.int8:
            output_low_value = -128.0
            output_high_value = 127.0
        elif zero_point_type == np.uint8:
            output_low_value = 0.0
            output_high_value = 255.0
        else:
            raise Error('Not expected type {} for zero point value in node {}'.format(
                zero_point_type, zerop.soft_get('name')))
        fake_quantize = create_op_with_const_inputs(graph, FakeQuantize, {3: float_array(output_low_value),
                                                                          4: float_array(output_high_value)},
                                                    {'levels': 256, 'name': node_name + '/FakeQuantize'})
        node.in_port(0).get_connection().set_destination(fake_quantize.in_port(0))

        # Calculate input_low value
        mul_low = create_op_with_const_inputs(graph, Mul, {1: float_array(output_low_value - zerop.value)},
                                              {'name': node_name + '/Mul/Low'})
        node.in_port(1).get_connection().set_destination(mul_low.in_port(0))

        # Calculate input_high value
        mul_high = create_op_with_const_inputs(graph, Mul, {1: float_array(output_high_value - zerop.value)},
                                              {'name': node_name + '/Mul/High'})
        mul_low.in_port(0).get_connection().add_destination(mul_high.in_port(0))

        cast = Cast(graph, {'dst_type': zero_point_type, 'name': node_name + '/Cast'}).create_node()
        rename_nodes([(node, node_name + '/TBD'), (cast, node_name)])
        fake_quantize.out_port(0).connect(cast.in_port(0))

        axis = node.soft_get('axis', node.id)
        if axis or axis != 1:
            data_shape_node = Shape(graph, {'name': node_name + '/Shape'}).create_node()
            fake_quantize.in_port(0).get_source().connect(data_shape_node.in_port(0))
            gather_node = create_op_with_const_inputs(graph, Gather, {1: int64_array([axis])},
                                                      {'name': node_name + '/Gather'})
            data_shape_node.out_port(0).connect(gather_node.in_port(0))

            rank_node = Rank(graph, {'name': node_name + '/Rank'}).create_node()
            fake_quantize.in_port(0).get_source().connect(rank_node.in_port(0))

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
            reshape_low_node = Reshape(graph, {'name': node_name + '/Reshape_low'}).create_node()
            mul_low.out_port(0).connect(reshape_low_node.in_port(0))
            scatter_elements_node.out_port(0).connect(reshape_low_node.in_port(1))
            reshape_low_node.out_port(0).connect(fake_quantize.in_port(1))

            reshape_high_node = Reshape(graph, {'name': node_name + '/Reshape_high'}).create_node()
            mul_high.out_port(0).connect(reshape_high_node.in_port(0))
            scatter_elements_node.out_port(0).connect(reshape_high_node.in_port(1))
            reshape_high_node.out_port(0).connect(fake_quantize.in_port(1))
        else:
            mul_low.out_port(0).connect(fake_quantize.in_port(1))
            mul_high.out_port(0).connect(fake_quantize.in_port(2))

        return [cast.id]
