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
from extensions.ops.elementwise import Mul
from extensions.ops.fakequantize import FakeQuantize
from mo.front.common.partial_infer.utils import float_array
from mo.front.common.replacement import FrontReplacementOp
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, Node, rename_nodes
from mo.ops.const import Const
from mo.utils.error import Error


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
        mul_low.out_port(0).connect(fake_quantize.in_port(1))

        # Calculate input_high value
        mul_high = create_op_with_const_inputs(graph, Mul, {1: float_array(output_high_value - zerop.value)},
                                              {'name': node_name + '/Mul/High'})
        mul_low.in_port(0).get_connection().add_destination(mul_high.in_port(0))
        mul_high.out_port(0).connect(fake_quantize.in_port(2))

        cast = Cast(graph, {'dst_type': zero_point_type, 'name': node_name + '/Cast'}).create_node()
        rename_nodes([(node, node_name + '/TBD'), (cast, node_name)])
        fake_quantize.out_port(0).connect(cast.in_port(0))

        return [cast.id]
