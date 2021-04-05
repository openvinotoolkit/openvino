# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.front.onnx.quantize_dequantize_linear import QuantizeDequantizeLinear
from extensions.ops.Cast import Cast
from extensions.ops.elementwise import Mul, Sub
from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Graph, Node, rename_nodes
from mo.middle.passes.convert_data_type import data_type_str_to_np


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
