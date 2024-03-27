# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.graph.perm_inputs import PermuteInputs
from openvino.tools.mo.middle.passes.convert_data_type import np_data_type_to_destination_type
from openvino.tools.mo.ops.op import Op


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
            'global_seed': 0,
            'op_seed': 0,
            'output_type': np.float32,
        }, attrs)

    def backend_attrs(self):
        return [('output_type', lambda node: np_data_type_to_destination_type(node.output_type)),
                'global_seed',
                'op_seed']

    @staticmethod
    def type_infer(node: Node):
        node.out_port(0).set_data_type(node['output_type'])

    @staticmethod
    def infer(node: Node):
        assert node.has_valid('output_type')

        node.out_port(0).data.set_shape(node.in_port(0).data.get_value())

        # We need to keep data type in data nodes corresponding to min and max values,
        # as min and max value type should be the same as output_type attribute of RandomUniform
        # operation. 'correct_data_type' attribute prevents changes of the data node type when
        # ir data type is not equal to data node type.
        node.in_node(1)['correct_data_type'] = True
        node.in_node(2)['correct_data_type'] = True

        PermuteInputs().set_input_permutation(node.in_node(0), node, 'output:0', 'shape')


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
            'in_ports_count': 1,
            'out_ports_count': 1,
            'global_seed': 0,
            'op_seed': 0,
            'output_type': np.float32,
        }, attrs)
