# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.utils import shape_array, dynamic_dimension
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.op import Op


class UniqueInternal(Op):
    op = 'Unique'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': self.op,
            'op': self.op,
            'version': 'opset10',
            'infer': self.infer,
            'in_ports_count': 2,
            'out_ports_count': 4
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return [
            'sorted',
            'index_element_type',
            'count_element_type',
        ]

    @staticmethod
    def infer(node: Node):
        input_shape = node.in_port(0).data.get_shape()
        if node.is_out_port_connected(0):
            if node.is_in_port_connected(1):
                axis = node.in_port(1).data.get_value()
                assert axis, "Unique must have constant axis."
                out_shape = input_shape.copy()
                out_shape[axis.item()] = dynamic_dimension
                node.out_port(0).data.set_shape(out_shape)
            else:
                # no axis, means flattening
                node.out_port(0).data.set_shape(
                    shape_array([dynamic_dimension]))
        if node.is_out_port_connected(1):
            node.out_port(1).data.set_shape(shape_array([dynamic_dimension]))
        if node.is_out_port_connected(2):
            node.out_port(2).data.set_shape(shape_array([dynamic_dimension]))
        if node.is_out_port_connected(3):
            node.out_port(3).data.set_shape(shape_array([dynamic_dimension]))
