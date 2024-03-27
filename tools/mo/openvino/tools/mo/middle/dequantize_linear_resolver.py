# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.ops.Cast import Cast
from openvino.tools.mo.ops.elementwise import Mul, Sub
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Graph, rename_nodes
from openvino.tools.mo.middle.passes.convert_data_type import data_type_str_to_np
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern
from openvino.tools.mo.ops.reshape import Reshape


class DequantizeLinearResolver(MiddleReplacementPattern):
    """
    Transformation result depends on the axis value.
    If the axis is not set or x_scale input is scalar or 1D tensor with one element then DequantizeLinear is
    replaced with the sub-graph which can be expressed with the following formula:
        y = (x - x_zero_point) * x_scale
    In other cases DequantizeLinear can be replace to formula with addition reshape x_zero_point and x_scale.
    Target shape for reshape depend on axis.
    """
    enabled = True
    graph_condition = [lambda graph: graph.graph['layout'] == 'NCHW']

    def run_after(self):
        from openvino.tools.mo.middle.quantize_dequantize_linear_resolver import QuantizeDequantizeLinearResolver
        return [QuantizeDequantizeLinearResolver]

    def find_and_replace_pattern(self, graph: Graph):
        for dequantize_node in graph.get_op_nodes(op='DequantizeLinear'):
            node_name = dequantize_node.soft_get('name', dequantize_node.id)
            axis = dequantize_node.soft_get('axis', None)
            scale_y_shape = dequantize_node.in_port(1).data.get_shape()
            model_data_type = data_type_str_to_np(graph.graph['cmd_params'].data_type)
            cast = Cast(graph, {'dst_type': model_data_type, 'name': node_name + '/Cast'}).create_node()
            dequantize_node.in_port(0).get_connection().set_destination(cast.in_port(0))
            mul = Mul(graph, {'can_be_fused': False}).create_node()

            is_second_port_connected = dequantize_node.is_in_port_connected(2)
            if is_second_port_connected:
                # its is necessary not to replace subrtract for pattern in offline transformations
                # See ConvertQuantizeDequantize transformation in ngraph
                sub = Sub(graph, {'name': node_name + '/Sub', 'zero_point_sub': True}).create_node()
                cast.out_port(0).connect(sub.in_port(0))
                dequantize_node.in_port(2).get_connection().set_destination(sub.in_port(1))
                sub.out_port(0).connect(mul.in_port(0))
            else:
                cast.out_port(0).connect(mul.in_port(0))

            dequantize_node.in_port(1).get_connection().set_destination(mul.in_port(1))
            dequantize_node.out_port(0).get_connection().set_source(mul.out_port(0))
            rename_nodes([(dequantize_node, node_name + '/TBD'), (mul, node_name)])

            assert scale_y_shape is not None
            if axis is not None and len(scale_y_shape) > 0 and scale_y_shape[0] > 1:
                input_shape = cast.in_port(0).data.get_shape()
                target_shape = np.ones(len(input_shape), np.int64)
                target_shape[axis] = input_shape[axis]

                mul_reshape = create_op_with_const_inputs(graph, Reshape, {1: int64_array(target_shape)},
                                                          {'name': node_name + '/Reshape/Mul'})
                mul.in_port(1).get_connection().set_destination(mul_reshape.in_port(0))
                mul_reshape.out_port(0).connect(mul.in_port(1))

                if is_second_port_connected:
                    sub_reshape = create_op_with_const_inputs(graph, Reshape, {1: int64_array(target_shape)},
                                                              {'name': node_name + '/Reshape/Sub'})
                    sub.in_port(1).get_connection().set_destination(sub_reshape.in_port(0))
                    sub_reshape.out_port(0).connect(sub.in_port(1))
