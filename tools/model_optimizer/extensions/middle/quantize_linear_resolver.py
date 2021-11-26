# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from extensions.ops.Cast import Cast
from extensions.ops.elementwise import Mul
from extensions.ops.fakequantize import FakeQuantize
from mo.front.common.partial_infer.utils import float_array, int64_array
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, rename_nodes
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.const import Const
from mo.ops.reshape import Reshape
from mo.utils.error import Error


class QuantizeLinearResolver(MiddleReplacementPattern):
    """
    Replaces QuantizeLinear with FakeQuantize
    Transformation result depends on the axis value.
    If the axis is not set or x_scale input is scalar or 1D tensor with one element then QuantizeLinear is
    replaced with the sub-graph which can be expressed with the following formula:
        QuantizeLinear -> FakeQuantize(input
                                       Mul(y_scale, Const(low_value))
                                       Mul(y_scale, Const(high_value))
                                       Const(low_value)
                                       Const(high_value))
        low_value and high_value depend on from y_zero_point type
    In other cases y_scale and y_zero_point can be transform with addition reshape.
    Target shape for y_scale and y_zero_point depend on axis value.
    """
    enabled = True
    graph_condition = [lambda graph: graph.graph['layout'] == 'NCHW']

    def run_after(self):
        from extensions.middle.quantize_fuses import MarkNodesToFuseUpToFakeQuantize
        return [MarkNodesToFuseUpToFakeQuantize]

    def find_and_replace_pattern(self, graph: Graph):
        for quantize_node in graph.get_op_nodes(op='QuantizeLinear'):
            node_name = quantize_node.soft_get('name', quantize_node.id)
            axis = quantize_node.soft_get('axis', None)
            scale_y_shape = quantize_node.in_port(1).data.get_shape()

            if quantize_node.is_in_port_connected(2):
                zerop = quantize_node.in_port(2).get_source().node
            else:
                zerop = Const(graph,
                              {'value': np.array(0, dtype=np.uint8), 'name': node_name + '/ZeroPoint'}).create_node()

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
            quantize_node.in_port(0).get_connection().set_destination(fake_quantize.in_port(0))

            # Calculate input_low value
            mul_low = create_op_with_const_inputs(graph, Mul, {1: float_array(output_low_value - zerop.value)},
                                                  {'name': node_name + '/Mul/Low'})
            quantize_node.in_port(1).get_connection().set_destination(mul_low.in_port(0))
            mul_low.out_port(0).connect(fake_quantize.in_port(1))

            # Calculate input_high value
            mul_high = create_op_with_const_inputs(graph, Mul, {1: float_array(output_high_value - zerop.value)},
                                                   {'name': node_name + '/Mul/High'})
            mul_low.in_port(0).get_connection().add_destination(mul_high.in_port(0))
            mul_high.out_port(0).connect(fake_quantize.in_port(2))

            cast = Cast(graph, {'dst_type': zero_point_type, 'name': node_name + '/Cast'}).create_node()
            fake_quantize.out_port(0).connect(cast.in_port(0))
            quantize_node.out_port(0).get_connection().set_source(cast.out_port(0))
            rename_nodes([(quantize_node, node_name + '/TBD'), (cast, node_name)])

            assert scale_y_shape is not None
            if axis is not None and len(scale_y_shape) > 0 and scale_y_shape[0] > 1:
                input_shape = fake_quantize.in_port(0).data.get_shape()
                target_shape = np.ones(len(input_shape), np.int)
                target_shape[axis] = input_shape[axis]
                mul_low_reshape = create_op_with_const_inputs(graph, Reshape, {1: int64_array(target_shape)},
                                                              {'name': node_name + '/Reshape/Mul/Low'})
                mul_high_reshape = create_op_with_const_inputs(graph, Reshape, {1: int64_array(target_shape)},
                                                               {'name': node_name + '/Reshape/Mul/high'})

                fake_quantize.in_port(1).get_connection().set_destination(mul_low_reshape.in_port(0))
                fake_quantize.in_port(2).get_connection().set_destination(mul_high_reshape.in_port(0))

                mul_low_reshape.out_port(0).connect(fake_quantize.in_port(1))
                mul_high_reshape.out_port(0).connect(fake_quantize.in_port(2))
