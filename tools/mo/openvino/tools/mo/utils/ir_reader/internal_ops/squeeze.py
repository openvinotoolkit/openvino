# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.ops.squeeze import Squeeze
from openvino.tools.mo.front.common.partial_infer.utils import shape_array, is_fully_defined


class SqueezeInternal(Squeeze):
    @staticmethod
    def infer(node: Node):
        if node.is_in_port_connected(1):
            axis_value = node.in_port(1).data.get_value()
            Squeeze.infer(node)
            # preserve initial axis value
            node.in_port(1).data.set_value(axis_value)
        else:
            # Squeeze without axes provided
            node_name = node.soft_get('name', node.id)
            input_shape = node.in_port(0).data.get_shape()
            assert is_fully_defined(
                input_shape), 'Squeeze dimensions are not defined for op "{}"'.format(node_name)
            output_shape = [s for s in shape_array(input_shape).tolist() if s != 1]
            node.out_port(0).data.set_shape(shape_array(output_shape))

