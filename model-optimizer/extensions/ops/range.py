"""
 Copyright (C) 2018-2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import logging as log

import numpy as np

from mo.graph.graph import Node, Graph
from mo.middle.passes.convert_data_type import np_data_type_to_destination_type
from mo.ops.op import Op
from mo.utils.error import Error


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

    >>> np.array([0.5], dtype=np.float32)
    array([0.5], dtype=float32)
    >>> np.arange(np.array([1], dtype=np.int32), np.array([5], dtype=np.int32), np.array([2], dtype=np.int32)).dtype
    dtype('int64')
    >>> np.arange(np.array([1], dtype=np.int32), np.array([5], dtype=np.int32), np.array([0.5], dtype=np.float32)).dtype
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

        assert start is not None and limit is not None and delta is not None, \
            'Range operation {} with dynamic inputs is not supported'.format(name)

        if not node.has_valid('output_type'):
            node['output_type'] = start.dtype

        node.out_port(0).data.set_value(np.arange(start, limit, delta, dtype=node['output_type']))
