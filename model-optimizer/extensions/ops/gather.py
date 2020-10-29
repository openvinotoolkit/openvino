"""
 Copyright (C) 2017-2020 Intel Corporation

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

import numpy as np

from mo.front.caffe.extractors.utils import get_canonical_axis_index
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node, Graph
from mo.ops.op import Op, PermuteAttrs


class Gather(Op):
    op = 'Gather'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': self.op,
            'type': self.op,
            'version': 'opset1',

            'infer': self.infer,

            'force_precision_in_ports': {1: 'int32', 2: 'int64'},

            'in_ports_count': 3,
            'out_ports_count': 1,
        }, attrs)

        assert 'axis' not in self.attrs, \
            'Use AttributedGather operation instead of Gather to create it with `axis` as a parameter'

    @staticmethod
    def infer(node: Node):
        name = node.soft_get('name', node.id)

        connected_in_ports = {idx: port for idx, port in node.in_ports().items() if not port.disconnected()}
        assert len(connected_in_ports) == 3 and 0 in connected_in_ports and 1 in connected_in_ports and \
               2 in connected_in_ports, "Gather should have 3 connected input port, but it doesn't for " \
                                        "node: `{}`. Ports: {}".format(name, connected_in_ports)

        data_shape = node.in_port(0).data.get_shape()
        assert data_shape is not None
        indices_shape = node.in_port(1).data.get_shape()
        assert indices_shape is not None
        axis = node.in_port(2).data.get_value()
        assert axis is not None
        axis = get_canonical_axis_index(data_shape, axis)

        # we import PermuteInputs locally because it uses Gather inside and we have recursive imports
        from mo.graph.perm_inputs import PermuteInputs
        PermuteInputs().set_input_permutation(node.in_node(1), node, 'input:0', 'axis')

        data_value = node.in_port(0).data.get_value()
        indices_value = node.in_port(1).data.get_value()
        if data_value is not None and indices_value is not None:
            node.out_port(0).data.set_value(np.array(np.take(data_value, int64_array(indices_value), axis),
                                                     dtype=data_value.dtype))
            return

        shape = np.concatenate((data_shape[:axis], indices_shape))
        if axis < len(data_shape) - 1:
            shape = np.concatenate((shape, data_shape[axis + 1:]))

        node.out_port(0).data.set_shape(int64_array(shape))


class AttributedGather(Op):
    op = 'AttributedGather'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': self.op,
            'type': 'Gather',

            'axis': 0,

            'infer': self.infer,

            'force_precision_in_ports': {1: 'int32'},

            'in_ports_count': 3,
            'out_ports_count': 1,
        }, attrs)

    def supported_attrs(self):
        return [
            'axis',
        ]

    @staticmethod
    def infer(node: Node):
        name = node.soft_get('name', node.id)

        connected_in_ports = {idx: port for idx, port in node.in_ports().items() if not port.disconnected()}
        assert len(connected_in_ports) == 2 and 0 in connected_in_ports and 1 in connected_in_ports, \
            "AttributedGather should have 2 connected input port, but it doesn't for node: `{}`. Ports: {}" \
            "".format(name, connected_in_ports)

        axis = node.soft_get('axis', None)
        assert axis is not None

        data_shape = node.in_port(0).data.get_shape()
        assert data_shape is not None
        indices_shape = node.in_port(1).data.get_shape()
        assert indices_shape is not None

        # Convert negative axis
        axis = get_canonical_axis_index(data_shape, axis)
        node.axis = axis

        PermuteAttrs.create_permute_attrs(node, attrs=[('axis', 'input:0')])

        data_value = node.in_port(0).data.get_value()
        indices_value = node.in_port(1).data.get_value()
        if data_value is not None and indices_value is not None:
            node.out_port(0).data.set_value(np.array(np.take(data_value, indices_value, axis), dtype=data_value.dtype))
            return

        shape = np.concatenate((data_shape[:axis], indices_shape))
        if axis < len(data_shape) - 1:
            shape = np.concatenate((shape, data_shape[axis + 1:]))

        node.out_port(0).data.set_shape(int64_array(shape))
