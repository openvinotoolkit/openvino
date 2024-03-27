# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.front.common.partial_infer.utils import shape_array, dynamic_dimension_value, is_fully_defined
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.middle.passes.convert_data_type import np_data_type_to_destination_type
from openvino.tools.mo.ops.op import Op
from openvino.tools.mo.utils.error import Error


class Range(Op):
    """
    Some notes on the automatic result data type infer. The tf.range does is differently than np.arange. Numpy
    by default creates array with elements of type int64 and float64, but TF does not widen data types and
    keep them int32 and float32.
    Compare:

    >>> tf.range(1, 5, 0.5)
    <tf.Tensor 'range_1:0' shape = (8,) dtype = float32>
    >>> tf.range(1, 5, 2)
    <tf.Tensor 'range_2:0' shape = (2,) dtype = int32>

    >>> mo_array([0.5], dtype=np.float32)
    array([0.5], dtype=float32)
    >>> np.arange(mo_array([1], dtype=np.int32), mo_array([5], dtype=np.int32), mo_array([2], dtype=np.int32)).dtype
    dtype('int64')
    >>> np.arange(mo_array([1], dtype=np.int32), mo_array([5], dtype=np.int32), mo_array([0.5], dtype=np.float32)).dtype
    dtype('float64')
    """
    op = 'Range'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': self.op,
            'op': self.op,

            'version': 'opset4',
            'infer': self.infer,
            'type_infer': self.type_infer,

            'in_ports_count': 3,
            'out_ports_count': 1,
        }
        super().__init__(graph, mandatory_props, attrs)

    def backend_attrs(self):
        version = self.get_opset()
        if version == 'opset4':
            return [
                ('output_type', lambda node: np_data_type_to_destination_type(node.output_type)),
            ]
        elif version == 'opset1':
            return []
        else:
            raise Error('Unknown opset version "{}"'.format(version))

    @staticmethod
    def type_infer(node: Node):
        node.out_port(0).set_data_type(node['output_type'])

    @staticmethod
    def infer(node: Node):
        name = node.soft_get('name', node.id)
        connected_input_ports = [in_port.idx for in_port in node.in_ports().values() if not in_port.disconnected()]
        assert len(connected_input_ports) == 3 and [0, 1, 2] == sorted(connected_input_ports), \
            'Range operation should have 3 inputs, {} found for {}'.format(len(connected_input_ports), name)

        start = node.in_port(0).data.get_value()
        limit = node.in_port(1).data.get_value()
        delta = node.in_port(2).data.get_value()

        for input in (start, limit, delta):
            if input is not None and not node.has_valid('output_type'):
                node['output_type'] = input.dtype

        if not is_fully_defined(start) or not is_fully_defined(limit) or not is_fully_defined(delta):
            node.out_port(0).data.set_shape(shape_array([dynamic_dimension_value]))
        else:
            node.out_port(0).data.set_value(np.arange(start, limit, delta, dtype=node['output_type']))
