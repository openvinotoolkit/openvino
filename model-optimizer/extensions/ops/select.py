"""
 Copyright (C) 2018-2020 Intel Corporation

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

from mo.front.common.partial_infer.utils import int64_array, broadcast_shape
from mo.graph.graph import Node, Graph
from mo.ops.op import Op


class Select(Op):
    op = 'Select'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'op': __class__.op,
            'type': __class__.op,
            'version': 'opset1',
            'in_ports_count': 3,
            'out_ports_count': 1,
            'infer': __class__.infer,
            'type_infer': __class__.type_infer,
        }
        super().__init__(graph, mandatory_props, attrs)

    @staticmethod
    def infer(node: Node):
        assert len([port for port in node.in_ports().values() if not port.disconnected()]) == 3, "Select operation must have 3 inputs:" \
                                          " \'condition\', \'then\' and \'else\' tensors"

        condition_value = node.in_port(0).data.get_value()
        resulting_tensors = [node.in_port(1).data.get_value(), node.in_port(2).data.get_value()]

        a_shape = node.in_port(1).data.get_shape()
        b_shape = node.in_port(2).data.get_shape()
        node.out_port(0).data.set_shape(broadcast_shape(a_shape, b_shape))
        np.broadcast(resulting_tensors[0], resulting_tensors[1])
        # Case with unknown condition
        if condition_value is not None:
            output_value = np.where(condition_value, resulting_tensors[0], resulting_tensors[1])
            if np.any(output_value == None):
                # If any element of output value is None that means that we use the value from 'then' or 'else' tensor
                # which is not defined, this means that we cannot perform value propagation.
                output_value = None

            if output_value is not None:
                node.out_port(0).data.set_value(np.array(output_value))

    @staticmethod
    def type_infer(node: Node):
        assert node.in_port(1).get_source().get_data_type() == node.in_port(2).get_source().get_data_type(), \
            'The data type of the second and the third inputs must be equal for the node {}'.format(node.name)
        node.out_port(0).set_data_type(node.in_port(1).get_source().get_data_type())
