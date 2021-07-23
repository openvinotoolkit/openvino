# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from mo.graph.graph import Graph, Node
from mo.middle.passes.convert_data_type import np_data_type_to_destination_type, np_data_type_to_str
from mo.ops.op import Op


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
            'in_ports_count': 3,
            'out_ports_count': 1,
            'type_infer': self.type_infer,
            'seed': 0,
            'seed2': 0,
            'output_type': np.float32,
            'force_precision_in_ports': {1: None, 2: None},
        }, attrs)

    def backend_attrs(self):
        return [('output_type', lambda node: np_data_type_to_destination_type(node.output_type)),
                'seed',
                'seed2']

    @staticmethod
    def type_infer(node: Node):
        node.out_port(0).set_data_type(node['output_type'])

    @staticmethod
    def infer(node: Node):
        assert node.has_valid('output_type')
        type_str = np_data_type_to_str(node['output_type'])
        node['force_precision_in_ports'] = {1: type_str, 2: type_str}

        out_shape = node.in_node(0).value
        node.out_node().shape = out_shape

        node.in_node(1)['correct_data_type'] = True
        node.in_node(2)['correct_data_type'] = True


class AttributedRandomUniform(Op):
    """ RandomUniform operation that generates a sequence of random values from uniform distribution.
        This operation uses the same semantics as RandomUniform but output shape, min value or max value
        can be specified as attribute.
        Shape is specified as attribute in ONNX. Min value and max value are specified as attributes
        in RandomUniformInt in TF.
    """
    op = 'AttributedRandomUniform'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': None,
            'op': self.op,
            'infer': None,
            'in_ports_count': 3,
            'out_ports_count': 1,
            'seed': 0,
            'seed2': 0,
            'output_type': np.float32,
        }, attrs)
