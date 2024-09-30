# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.middle.passes.convert_data_type import np_data_type_to_destination_type
from openvino.tools.mo.ops.op import Op
from openvino.tools.mo.front.common.partial_infer.utils import dynamic_dimension, shape_array
from openvino.tools.mo.front.common.partial_infer.utils import is_fully_defined


class Eye(Op):
    """
    Eye operation that generates shift matrix or a batch of matrices.
    """
    op = 'Eye'
    enabled = False
    in_ports_count = 4

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': self.op,
            'op': self.op,
            'version': 'opset9',
            'infer': self.infer,
            'in_ports_count': 4,
            'out_ports_count': 1,
            'type_infer': self.type_infer,
            'output_type': np.float32,
        }, attrs)

    def backend_attrs(self):
        return [('output_type', lambda node: np_data_type_to_destination_type(node.output_type))]

    @staticmethod
    def type_infer(node: Node):
        node.out_port(0).set_data_type(node['output_type'])

    @staticmethod
    def infer(node: Node):
        assert node.has_valid('output_type')

        connected_in_ports = {idx: port for idx, port in node.in_ports().items() if not port.disconnected()}
        assert len(connected_in_ports) >= 3 and all(p in connected_in_ports for p in [0, 1, 2]), \
            "Eye should have at least 3 connected input port." \
            "Got ports: `{}`.".format(connected_in_ports)

        num_rows_port = 0
        num_columns_port = 1
        diagonal_index_port = 2
        batch_shape_port = 3

        num_rows_shape = node.in_port(num_rows_port).data.get_shape()
        assert len(num_rows_shape) <= 1, \
            '"num_rows" should be 1D tensor or scalar. Got: '.format(len(num_rows_shape))
        num_rows = node.in_port(num_rows_port).data.get_value()
        if num_rows is None:
            num_rows = dynamic_dimension
        else:
            num_rows = np.array(num_rows).item()

        num_columns_shape = node.in_port(num_columns_port).data.get_shape()
        assert len(num_columns_shape) <= 1, \
            '"num_columns" should be 1D tensor or scalar. Got: '.format(len(num_columns_shape))
        num_columns = node.in_port(num_columns_port).data.get_value()
        if num_columns is None:
            num_columns = dynamic_dimension
        else:
            num_columns = np.array(num_columns).item()

        diagonal_index_shape = node.in_port(diagonal_index_port).data.get_shape()
        assert len(diagonal_index_shape) <= 1, \
            '"diagonal_index" should be 1D tensor or scalar. Got: '.format(len(diagonal_index_shape))
        diagonal_index = node.in_port(diagonal_index_port).data.get_value()

        if batch_shape_port in connected_in_ports:
            batch_shape_shape = node.in_port(batch_shape_port).data.get_shape()
            assert len(batch_shape_shape) == 1, \
                '"batch_shape" should be 1D tensor. Got: '.format(len(batch_shape_shape))
            batch_shape = node.in_port(batch_shape_port).data.get_value()
            if batch_shape is None:
                batch_shape = [dynamic_dimension] * batch_shape_shape[0]
        else:
            batch_shape = []

        output_shape = [*batch_shape, num_rows, num_columns]
        node.out_port(0).data.set_shape(output_shape)

        if is_fully_defined(output_shape) and diagonal_index is not None:
            tile_shape = [*batch_shape, 1, 1]
            one_matrix = np.eye(num_rows, M=num_columns, k=np.array(diagonal_index).item(), dtype=node.output_type)
            output_value = np.tile(one_matrix, tile_shape)
            node.out_port(0).data.set_value(shape_array(output_value))


class TFEye(Op):
    """ Eye operation that that generates shift matrix or a batch of matrices.
        Eye operation from TensorFlow has three inputs: row number, column number and batch shape
    """
    op = 'TFEye'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': None,
            'op': self.op,
            'infer': None,
            'in_ports_count': 3,
            'out_ports_count': 1,
            'output_type': np.float32,
        }, attrs)
