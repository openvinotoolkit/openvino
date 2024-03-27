# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import dynamic_dimension
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.op import Op, PermuteAttrs
from openvino.tools.mo.utils.error import Error


class GatherElements(Op):
    op = 'GatherElements'

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': self.op,
            'type': self.op,
            'version': 'opset6',
            'infer': self.infer,
            'in_ports_count': 2,
            'out_ports_count': 1,
            'axis': 0,
        }, attrs)

    def backend_attrs(self):
        return ['axis']

    @staticmethod
    def infer(node: Node):
        data_shape = node.in_port(0).data.get_shape()
        indices_shape = node.in_port(1).data.get_shape()
        axis = node.axis
        data_rank = len(data_shape)

        assert data_rank >= 1, 'data_rank must be >= 1'
        assert data_rank == len(indices_shape), 'data and indices inputs for node {} must be of the ' \
                                                'same rank. Instead got {} and {}'. \
            format(node.name, data_rank, len(indices_shape))
        assert -data_rank <= axis < data_rank, 'axis for node {0} must be within interval ' \
                                               '[-{1},  {1} - 1]. Instead got: axis={2}'. \
            format(node.name, data_rank, axis)
        if axis < 0:
            axis += data_rank
        out_shape = indices_shape.copy()
        for idx, (data_sz, ind_sz) in enumerate(zip(data_shape, indices_shape)):
            out_shape[idx] = ind_sz if ind_sz is not dynamic_dimension or idx == axis else data_sz
            if idx != axis and data_sz != ind_sz:
                raise Error('Sizes along axis {} for node {} do not match. data and indices must have '
                            'equal size along all axes except for axis {}'.format(idx, node.name, axis))

        data = node.in_port(0).data.get_value()
        indices = node.in_port(1).data.get_value()

        if data is not None and indices is not None:
            out_value = np.empty(indices_shape, dtype=data.dtype)
            for idx in np.ndindex(*indices_shape):
                data_idx = list(idx)
                data_idx[node.axis] = indices[idx]
                out_value[idx] = data[tuple(data_idx)]
            node.out_port(0).data.set_value(out_value)
        else:
            node.out_port(0).data.set_shape(out_shape)

        PermuteAttrs.create_permute_attrs(node, attrs=[('axis', 'input:0')])
