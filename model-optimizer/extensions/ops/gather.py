# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from mo.front.caffe.extractors.utils import get_canonical_axis_index
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node, Graph
from mo.ops.op import Op, PermuteAttrs
from mo.utils.error import Error


class Gather(Op):
    op = 'Gather'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': self.op,
            'type': self.op,
            'version': 'opset7',
            'batch_dims': 0,
            'infer': self.infer,
            'force_precision_in_ports': {1: 'int32', 2: 'int64'},
            'in_ports_count': 3,
            'out_ports_count': 1,
        }, attrs)

        assert 'axis' not in self.attrs, \
            'Use AttributedGather operation instead of Gather to create it with `axis` as a parameter'

    def backend_attrs(self):
        version = self.get_opset()
        if version == 'opset7':
            return [('batch_dims', lambda node: Gather.get_batch_dims(node))]
        elif version == 'opset1':
            return []
        else:
            raise Error('Unsupported operation opset version "{}"'.format(version))

    @staticmethod
    def get_axis(node: Node):
        axis_value = node.in_port(2).data.get_value()
        assert axis_value is not None, 'Cannot define axis value for operation {}'.format(node.soft_get('name', node.id))
        if axis_value >= 0:
            return axis_value
        else:
            data_rank = len(node.in_port(0).data.get_shape())
            return axis_value + data_rank

    @staticmethod
    def get_batch_dims(node: Node):
        indicies_rank = len(node.in_port(1).data.get_shape())
        return node.batch_dims + indicies_rank if node.batch_dims < 0 else node.batch_dims

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
        assert node.in_port(2).data.get_value() is not None
        axis = Gather.get_axis(node)

        # we import PermuteInputs locally because it uses Gather inside and we have recursive imports
        from mo.graph.perm_inputs import PermuteInputs
        PermuteInputs().set_input_permutation(node.in_node(1), node, 'input:0', 'axis')

        batch_dims = Gather.get_batch_dims(node)
        batch_dims_range = indices_shape[:batch_dims]
        out_shape = np.concatenate((data_shape[:axis], indices_shape[batch_dims:], data_shape[axis + 1:]))

        data_value = node.in_port(0).data.get_value()
        indices_value = node.in_port(1).data.get_value()
        if data_value is not None and indices_value is not None:
            if not batch_dims:
                node.out_port(0).data.set_value(np.take(data_value, indices_value, axis))
                return

            out_value = np.empty(out_shape)
            for batch_idx in np.ndindex(tuple(batch_dims_range)):
                out_value[batch_idx] = np.take(data_value[batch_idx], indices_value[batch_idx], axis - batch_dims)
            node.out_port(0).data.set_value(out_value)
            return

        node.out_port(0).data.set_shape(int64_array(out_shape))


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

            'in_ports_count': 2,
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
