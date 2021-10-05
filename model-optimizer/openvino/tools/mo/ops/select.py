# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import is_fully_defined
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.op import Op
from openvino.tools.mo.utils.broadcasting import bi_directional_shape_broadcasting, bi_directional_broadcasting


class Select(Op):
    op = 'Select'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'op': self.op,
            'type': self.op,
            'version': 'opset1',
            'in_ports_count': 3,
            'out_ports_count': 1,
            'infer': self.infer,
            'type_infer': self.type_infer,
            'auto_broadcast': 'numpy'
        }
        super().__init__(graph, mandatory_props, attrs)

    def backend_attrs(self):
        return ['auto_broadcast']

    @staticmethod
    def infer(node: Node):
        node_name = node.soft_get('name', node.id)
        assert len([port for port in node.in_ports().values() if not port.disconnected()]) == 3, \
            "Select operation must have 3 inputs: 'condition', 'then' and 'else' tensors for node {}".format(node_name)

        condition_value = node.in_port(0).data.get_value()
        resulting_tensors = [node.in_port(1).data.get_value(), node.in_port(2).data.get_value()]

        a_shape = node.in_port(1).data.get_shape()
        b_shape = node.in_port(2).data.get_shape()
        output_shape = bi_directional_shape_broadcasting(a_shape, b_shape)
        assert output_shape is not None, 'Input shapes for node {} are not broadcast-able'.format(node_name)
        node.out_port(0).data.set_shape(output_shape)

        if condition_value is not None:
            if resulting_tensors[0] is not None:
                resulting_tensors[0] = bi_directional_broadcasting(resulting_tensors[0], b_shape)
            if resulting_tensors[1] is not None:
                resulting_tensors[1] = bi_directional_broadcasting(resulting_tensors[1], a_shape)
            condition_value = bi_directional_broadcasting(condition_value, output_shape)

            output_value = np.ma.where(condition_value, resulting_tensors[0], resulting_tensors[1])
            if condition_value.size != 1:
                if np.any(output_value == None):
                    # If any element of output value is None that means that we use the value from the 'then' or the
                    # 'else' tensor which is not defined, this means that we cannot perform value propagation.
                    output_value = None
            else:
                output_value = output_value.astype(resulting_tensors[not np.bool(condition_value.item(0))].dtype)

            if output_value is not None:
                node.out_port(0).data.set_value(output_value)

    @staticmethod
    def type_infer(node: Node):
        assert node.in_port(1).get_source().get_data_type() == node.in_port(2).get_source().get_data_type(), \
            'The data type of the second and the third inputs must be equal for the node {}'.format(node.name)
        node.out_port(0).set_data_type(node.in_port(1).get_source().get_data_type())
