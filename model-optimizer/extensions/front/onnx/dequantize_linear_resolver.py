"""
 Copyright (C) 2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import numpy as np

from extensions.front.onnx.quantize_dequantize_linear import QuantizeDequantizeLinear
from extensions.ops.Cast import Cast
from extensions.ops.elementwise import Mul, Sub
from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementOp
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, Node, rename_nodes
from mo.ops.broadcast import Broadcast
from mo.ops.shape import Shape


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
        cast = Cast(graph, {'dst_type': np.float32, 'name': node_name + '/Cast'}).create_node()
        node.in_port(0).get_connection().set_destination(cast.in_port(0))
        mul = Mul(graph, {}).create_node()

        if node.has_valid('axis'):
            shape = Shape(graph, {'name': node_name + '/Shape'}).create_node()
            cast.in_port(0).get_connection().add_destination(shape.in_port(0))

        if node.is_in_port_connected(2):
            sub = Sub(graph, {'name': node_name + '/Sub'}).create_node()
            cast.out_port(0).connect(sub.in_port(0))
            if node.has_valid('axis'):
                bc2 = create_op_with_const_inputs(graph, Broadcast, {2: int64_array(node['axis'])},
                                                  {'mode': 'explicit', 'name': node_name + '/BC2'})
                node.in_port(2).get_connection().set_destination(bc2.in_port(0))
                shape.out_port(0).connect(bc2.in_port(1))
                bc2.out_port(0).connect(sub.in_port(1))
            else:
                node.in_port(2).get_connection().set_destination(sub.in_port(1))
            sub.out_port(0).connect(mul.in_port(0))
        else:
            cast.out_port(0).connect(mul.in_port(0))
        if node.has_valid('axis'):
            bc1 = create_op_with_const_inputs(graph, Broadcast, {2: int64_array(node['axis'])},
                                              {'mode': 'explicit', 'name': node_name + '/BC1'})
            node.in_port(1).get_connection().set_destination(bc1.in_port(0))
            shape.out_port(0).get_connection().add_destination(bc1.in_port(1))
            bc1.out_port(0).connect(mul.in_port(1))
        else:
            node.in_port(1).get_connection().set_destination(mul.in_port(1))
        rename_nodes([(node, node_name + '/TBD'), (mul, node_name)])

        return [mul.id]
