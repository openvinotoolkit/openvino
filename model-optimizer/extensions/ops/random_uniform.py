# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.graph.graph import Graph, Node
from mo.ops.op import Op
from mo.middle.passes.convert_data_type import np_data_type_to_destination_type


class RandomUniform(Op):
    """
    RandomUniform operation that generates a sequence of random values from uniform distribution.
    """
    op = 'RandomUniform'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': self.op,
            'op': self.op,
            'version': 'opset8',
            'infer': self.infer,
            'in_ports_count': 1,
            'out_ports_count': 1,
            'type_infer': self.type_infer,
        }, attrs)

    def backend_attrs(self):
        return [('output_type', lambda node: np_data_type_to_destination_type(node.output_type)), 'seed', 'seed2']

    def supported_attrs(self):
        return [('output_type', lambda node: np_data_type_to_destination_type(node.output_type)), 'seed', 'seed2']

    @staticmethod
    def type_infer(node: Node):
        node.out_port(0).set_data_type(node['output_type'])

    @staticmethod
    def infer(node: Node):
        out_shape = node.in_node(0).value
        node.out_node().shape = out_shape
