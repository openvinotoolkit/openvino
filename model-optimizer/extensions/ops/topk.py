# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from mo.graph.graph import Graph
from mo.middle.passes.convert_data_type import np_data_type_to_destination_type
from mo.ops.op import Op, PermuteAttrs
from mo.utils.error import Error


class TopK(Op):
    op = 'TopK'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': __class__.op,
            'op': __class__.op,
            'version': 'opset3',
            'infer': self.infer,
            'type_infer': self.type_infer,

            'index_element_type': np.int32,
            'axis': None,
            'mode': 'max',
            'sort': 'none',
            'force_precision_in_ports': {
                1: 'int32'},
            'in_ports_count': 3,
            'out_ports_count': 2,
        }, attrs)

    def backend_attrs(self):
        version = self.get_opset()
        if version == 'opset3':
            return ['axis', 'mode', 'sort',
                    ('index_element_type', lambda node: np_data_type_to_destination_type(node.index_element_type))]
        elif version == 'opset1':
            return ['axis', 'mode', 'sort']
        else:
            raise Error('Unknown opset version "{}"'.format(version))

    @staticmethod
    def infer(node):
        in_ports = node.in_ports()
        connected_ports = [port for port in in_ports.values() if not port.disconnected()]
        assert len(connected_ports) == 2, 'The number of inputs to the TopK layer name "{}" must be equal to 2.' \
                                          ''.format(node.soft_get('name'))

        k = node.in_port(1).data.get_value()
        if k is None:
            raise Error('The value defining number of output elements for layer "{}" is not defined'
                        ''.format(node.soft_get('name')))
        assert node.has_valid('axis'), 'The "axis" attribute is not defined for node {}'.format(node.name)

        input_shape = node.in_port(0).data.get_shape()
        node.axis = len(input_shape) + node.axis if node.axis < 0 else node.axis
        output_shape = input_shape.copy()
        output_shape[node.axis] = k

        PermuteAttrs.create_permute_attrs(node, attrs=[('axis', 'input:0')])

        # setting shape and value if applicable
        if not node.out_port(0).disconnected():
            node.out_port(0).data.set_shape(output_shape)
        if not node.out_port(1).disconnected():
            node.out_port(1).data.set_shape(output_shape)
        if node.in_port(0).data.get_value() is not None:
            # TODO implement value propagation
            pass

    @staticmethod
    def type_infer(node):
        node.out_port(0).set_data_type(node.in_port(0).get_data_type())
        if node.get_opset() == 'opset3':
            node.out_port(1).set_data_type(node.index_element_type)
        else:
            node.out_port(1).set_data_type(np.int32)
