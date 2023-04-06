# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.ops.op import Op


class ReadValue(Op):
    op = 'ReadValue'

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': self.op,
            'op': self.op,
            'version': 'opset6',
            'infer': self.infer,
            'type_infer': self.type_infer,
            'in_ports_count': 1,
            'out_ports_count': 1,
        }, attrs)

    def backend_attrs(self):
        return ['variable_id']

    @staticmethod
    def type_infer(node: Node):
        node.out_port(0).set_data_type(node.in_port(0).get_data_type())

    @staticmethod
    def infer(node: Node):
        assert node.has_valid('variable_id'), \
            "There is no required attribute variable_id in ReadValue op with name " + node.id
        node.out_port(0).data.set_shape(node.in_port(0).data.get_shape())
