# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.caffe.extractors.utils import get_canonical_axis_index
from openvino.tools.mo.front.common.partial_infer.utils import mo_array, int64_array, is_fully_defined, shape_array, \
    dynamic_dimension_value
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.op import Op, PermuteAttrs
from openvino.tools.mo.utils.error import Error


class Gather(Op):
    op = 'Gather'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': self.op,
            'type': self.op,
            'version': 'opset8',
            'batch_dims': 0,
            'reinterp_shape': True,
            'infer': self.infer,
            'reverse_infer': self.reverse_infer,
            'force_precision_in_ports': {1: 'int32', 2: 'int64'},
            'in_ports_count': 3,
            'out_ports_count': 1,
        }, attrs)

        assert 'axis' not in self.attrs, \
            'Use AttributedGather operation instead of Gather to create it with `axis` as a parameter'

    def backend_attrs(self):
        version = self.get_opset()
        if version in ['opset7', 'opset8']:
            return ['batch_dims']
        elif version == 'opset1':
            return []
        else:
            raise Error('Unsupported operation opset version "{}"'.format(version))

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

        # axis of Gather could be accepted as both scalar and 1D tensor
        if isinstance(axis, np.ndarray):
            axis = axis.item()
        assert axis is not None, 'axis input is undefined'

        assert -len(data_shape) <= axis < len(data_shape), \
            'axis must be within interval [-data_rank, data_rank). Instead got axis = {}, data_rank = {} '.\
            format(axis, len(data_shape))

        batch_dims = node.batch_dims
        assert -len(indices_shape) <= batch_dims <= len(indices_shape), \
            'batch_dims must be within interval [-indices_rank, indices_rank]. Instead got batch_dims = {}, ' \
            'indices_rank = {} '.format(batch_dims, len(indices_shape))

        # normalize to positive values
        axis = axis + len(data_shape) if axis < 0 else axis
        batch_dims = batch_dims + len(indices_shape) if batch_dims < 0 else batch_dims

        assert np.ma.allequal(data_shape[:batch_dims], indices_shape[:batch_dims]), \
            'data and indices inputs must have equal first dimensions until batch_dims'

        assert batch_dims <= axis, \
            'normalized batch_dims must be <= axis. Instead got batch_dims = {}, axis = {}'.format(axis, batch_dims)

        # we import PermuteInputs locally because it uses Gather inside and we have recursive imports
        from openvino.tools.mo.graph.perm_inputs import PermuteInputs
        PermuteInputs().set_input_permutation(node.in_node(2), node, 'input:0', 'axis')

        batch_dims_range = indices_shape[:batch_dims]
        out_shape = np.concatenate((data_shape[:axis], indices_shape[batch_dims:], data_shape[axis + 1:]))

        data_value = node.in_port(0).data.get_value()
        indices_value = node.in_port(1).data.get_value()
        if data_value is not None and indices_value is not None and is_fully_defined(indices_value):
            indices_value = int64_array(indices_value)
            if batch_dims == 0:
                node.out_port(0).data.set_value(np.ma.take(data_value, indices_value, axis))
            else:
                out_value = np.empty(out_shape)
                for batch_idx in np.ndindex(tuple(batch_dims_range)):
                    out_value[batch_idx] = np.ma.take(data_value[batch_idx], indices_value[batch_idx],
                                                      axis - batch_dims)
                node.out_port(0).data.set_value(out_value)
        else:
            node.out_port(0).data.set_shape(out_shape)

    @staticmethod
    def reverse_infer(node: Node):
        out_shape = node.out_port(0).data.get_shape()
        data_shape = node.in_port(0).data.get_shape()
        indices_shape = node.in_port(1).data.get_shape()
        batch_dims = node.batch_dims
        batch_dims = batch_dims + len(indices_shape) if batch_dims < 0 else batch_dims

        axis = node.in_port(2).data.get_value()
        # axis of Gather could be accepted as both scalar and 1D tensor
        if isinstance(axis, np.ndarray):
            axis = axis.item()
        assert axis is not None, 'axis input is undefined'

        # we can deduce data or indices partial shapes from output shape calculation formula
        # out_shape = Concat(data_shape[:axis], indices_shape[batch_dims:batch_dims + indices_rank], data_shape[axis + 1:])

        # data partial shape is unknown
        if out_shape is not None and data_shape is None and indices_shape is not None:
            out_rank = len(out_shape)
            indices_rank = len(indices_shape)

            deduced_data_shape = out_shape.tolist(dynamic_dimension_value)
            for i in range(indices_rank):
                deduced_data_shape.pop(axis)
            deduced_data_shape.insert(axis, dynamic_dimension_value)
            node.in_port(0).data.set_shape(shape_array(deduced_data_shape))

        # indices partial shape is unknown
        if out_shape is not None and indices_shape is None and data_shape is not None:
            out_rank = len(out_shape)
            data_rank = len(data_shape)
            indices_rank = out_rank + 1 - data_rank + batch_dims

            indices_shape = out_shape[axis:axis + indices_rank]
            node.in_port(1).data.set_shape(indices_shape)


class AttributedGather(Op):
    op = 'AttributedGather'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': self.op,
            'type': 'Gather',

            'axis': 0,
            'reinterp_shape': True,
            'infer': self.infer,
            # reverse_infer is not needed since is replaced by Gather on the front (AttributedGatherNormalizer)

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
            node.out_port(0).data.set_value(mo_array(np.take(data_value, indices_value, axis), dtype=data_value.dtype))
            return

        shape = np.concatenate((data_shape[:axis], indices_shape))
        if axis < len(data_shape) - 1:
            shape = np.concatenate((shape, data_shape[axis + 1:]))

        node.out_port(0).data.set_shape(int64_array(shape))
