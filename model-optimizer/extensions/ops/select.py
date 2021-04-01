# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

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
            'auto_broadcast': 'numpy'
        }
        super().__init__(graph, mandatory_props, attrs)

    def backend_attrs(self):
        return ['auto_broadcast']

    @staticmethod
    def infer(node: Node):
        assert len([port for port in node.in_ports().values() if not port.disconnected()]) == 3, "Select operation must have 3 inputs:" \
                                          " \'condition\', \'then\' and \'else\' tensors"

        condition_value = node.in_port(0).data.get_value()
        resulting_tensors = [node.in_port(1).data.get_value(), node.in_port(2).data.get_value()]

        a_shape = node.in_port(1).data.get_shape()
        b_shape = node.in_port(2).data.get_shape()
        node.out_port(0).data.set_shape(broadcast_shape(a_shape, b_shape))
        # Case with unknown condition
        if condition_value is not None:
            output_value = np.where(condition_value, resulting_tensors[0], resulting_tensors[1])
            if condition_value.size != 1:
                if np.any(output_value == None):
                    # If any element of output value is None that means that we use the value from 'then' or 'else' tensor
                    # which is not defined, this means that we cannot perform value propagation.
                    output_value = None
            else:
                output_value = np.array(output_value, dtype=resulting_tensors[not np.bool(condition_value.item(0))].dtype)

            if output_value is not None:
                node.out_port(0).data.set_value(np.array(output_value))

    @staticmethod
    def type_infer(node: Node):
        assert node.in_port(1).get_source().get_data_type() == node.in_port(2).get_source().get_data_type(), \
            'The data type of the second and the third inputs must be equal for the node {}'.format(node.name)
        node.out_port(0).set_data_type(node.in_port(1).get_source().get_data_type())
