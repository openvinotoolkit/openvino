# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from extensions.front.onnx.quantize_dequantize_linear import QuantizeDequantizeLinear
from extensions.ops.Cast import Cast
from extensions.ops.elementwise import GreaterEqual
from extensions.ops.elementwise import Mul
from extensions.ops.fakequantize import FakeQuantize
from extensions.ops.gather import Gather
from extensions.ops.range import Range
from extensions.ops.scatter import ScatterElementsUpdate
from mo.front.common.partial_infer.utils import float_array, int64_array
from mo.front.common.replacement import FrontReplacementOp
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, Node, rename_nodes
from mo.ops.const import Const
from mo.ops.reshape import Reshape
from mo.ops.shape import Shape
from mo.ops.squeeze import Squeeze
from mo.utils.error import Error


def quantization_target_shape_subgraph(graph: Graph, node_name: str, input_node: Node, axis: int):
    """
    Generate subgraph to calculate target shape by axis for QuantizeLinear/DequantizeLinear ops.
    """
    data_shape_node = Shape(graph, {'name': node_name + '/Shape'}).create_node()
    input_node.out_port(0).connect(data_shape_node.in_port(0))
    gather_node = create_op_with_const_inputs(graph, Gather, {1: int64_array([axis]),
                                                              2: int64_array(0)},
                                              {'name': node_name + '/Gather'})
    data_shape_node.out_port(0).connect(gather_node.in_port(0))

    rank_node = Shape(graph, {'name': node_name + '/Rank'}).create_node()
    data_shape_node.out_port(0).connect(rank_node.in_port(0))
    rank_node_0d = create_op_with_const_inputs(graph, Squeeze, {1: int64_array(0)},
                                               {'name': node_name + '/0d_rank_of'})
    rank_node.out_port(0).connect(rank_node_0d.in_port(0))

    range_node = create_op_with_const_inputs(graph, Range, {0: int64_array(0),
                                                            2: int64_array(1)},
                                             {'name': node_name + '/Range'})
    rank_node_0d.out_port(0).connect(range_node.in_port(1))
    greater_equal_node = create_op_with_const_inputs(graph, GreaterEqual, {1: int64_array([0])},
                                                     {'name': node_name + '/GreaterEqual'})
    range_node.out_port(0).connect(greater_equal_node.in_port(0))

    scatter_elements_node = create_op_with_const_inputs(graph, ScatterElementsUpdate, {1: int64_array([axis]),
                                                                                       3: int64_array(0)},
                                                        {'name': node_name + '/ScatterElements'})

    gather_node.out_port(0).connect(scatter_elements_node.in_port(2))

    cast_node = Cast(graph, {'dst_type': np.int64, 'name': node_name + '/Cast'}).create_node()
    greater_equal_node.out_port(0).connect(cast_node.in_port(0))
    cast_node.out_port(0).connect(scatter_elements_node.in_port(0))

    return scatter_elements_node


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
        axis = node.soft_get('axis', None)

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
                                                                         {'levels': 256, 'axis': axis,
                                                                          'name': node_name + '/FakeQuantize'})
        node.in_port(0).get_connection().set_destination(fake_quantize.in_port(0))

        # Calculate input_low value
        mul_low = create_op_with_const_inputs(graph, Mul, {1: float_array(output_low_value - zerop.value)},
                                              {'name': node_name + '/Mul/Low'})
        node.in_port(1).get_connection().set_destination(mul_low.in_port(0))
        mul_low.out_port(0).connect(fake_quantize.in_port(1))

        # Calculate input_high value
        mul_high = create_op_with_const_inputs(graph, Mul, {1: float_array(output_high_value - zerop.value)},
                                               {'name': node_name + '/Mul/High'})
        mul_low.in_port(0).get_connection().add_destination(mul_high.in_port(0))
        mul_high.out_port(0).connect(fake_quantize.in_port(2))

        cast = Cast(graph, {'dst_type': zero_point_type, 'name': node_name + '/Cast'}).create_node()
        fake_quantize.out_port(0).connect(cast.in_port(0))

        if axis:
            target_shape_node = quantization_target_shape_subgraph(graph, node_name,
                                                                   fake_quantize.in_port(0).get_source().node, axis)
            scale_low = Reshape(graph, {'name': node_name + '/Reshape_scale_low'}).create_node()
            target_shape_node.out_port(0).connect(scale_low.in_port(1))

            scale_high = Reshape(graph, {'name': node_name + '/Reshape_scale_high'}).create_node()
            target_shape_node.out_port(0).connect(scale_high.in_port(1))

            fake_quantize.in_port(1).get_connection().set_destination(scale_low.in_port(0))
            fake_quantize.in_port(2).get_connection().set_destination(scale_high.in_port(0))

            scale_low.out_port(0).connect(fake_quantize.in_port(1))
            scale_high.out_port(0).connect(fake_quantize.in_port(2))

        rename_nodes([(node, node_name + '/TBD'), (cast, node_name)])
        return [cast.id]
