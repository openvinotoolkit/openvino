# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import shape_array, dynamic_dimension_value
from openvino.tools.mo.front.extractor import bool_to_str
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.middle.passes.convert_data_type import np_data_type_to_destination_type
from openvino.tools.mo.ops.op import Op
from openvino.tools.mo.utils.error import Error


class NMSRotated(Op):
    op = 'NMSRotated'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': self.op,
            'op': self.op,
            'version': 'opset13',

            'infer': self.infer,
            'type_infer': self.type_infer,

            'sort_result_descending': True,
            'output_type': np.int64,
            'clockwise': True,

            'in_ports_count': 5,
            'out_ports_count': 3,
        }
        super().__init__(graph, mandatory_props, attrs)

    def backend_attrs(self):
        return [('sort_result_descending', lambda node: bool_to_str(node, 'sort_result_descending')),
                'output_type',
                ('clockwise', lambda node: bool_to_str(node, 'clockwise'))]

    def supported_attrs(self):
        return [
            'sort_result_descending',
            'output_type',
            'clockwise',
        ]

    @staticmethod
    def infer(node: Node):
        num_of_inputs = len(node.in_ports())
        opset = node.get_opset()
        required_num_inputs = 5
        input_msg_fmt = 'NMSRotated node {} from {} must have {} inputs'
        node_name = node.soft_get('name', node.id)
        inputs_msg = input_msg_fmt.format(
            node_name, opset, required_num_inputs)
        assert num_of_inputs == required_num_inputs, inputs_msg

        node.out_port(0).data.set_shape(
            shape_array([dynamic_dimension_value, 3]))
        num_of_outputs = len(
            [port for port in node.out_ports().values() if not port.disconnected()])
        if num_of_outputs >= 2 and node.has_port('out', 1):
            node.out_port(1).data.set_shape(
                shape_array([dynamic_dimension_value, 3]))
        if num_of_outputs >= 3 and node.has_port('out', 2):
            node.out_port(2).data.set_shape(shape_array([1]))

    @staticmethod
    def type_infer(node: Node):
        node.out_port(1).set_data_type(np.float32)
        if node.has_valid('output_type') and node['output_type'].lower() == 'i32':
            node.out_port(0).set_data_type(np.int32)
            node.out_port(2).set_data_type(np.int32)
        else:
            node.out_port(0).set_data_type(np.int64)
            node.out_port(2).set_data_type(np.int64)
