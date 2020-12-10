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

from mo.graph.graph import Node, Graph
from mo.ops.op import Op, PermuteAttrs


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

        assert data_rank == len(indices_shape), f'data and indices inputs for node {node.name} must be of the ' \
                                                f'same rank. Got {data_rank} and {len(indices_shape)}'
        assert -data_rank < axis < data_rank, f'axis for node {node.name} must be within interval ' \
                                              f'[{-data_rank},  {data_rank - 1}]. Instead axis={axis}'
        if data_rank < 0:
            axis += data_rank

        for idx, (data_sz, ind_sz) in enumerate(zip(data_shape, indices_shape)):
            if idx != axis and data_sz != ind_sz:
                raise ValueError(f'Sizes along axis {idx} for node {node.name} do not match. '
                                 f'data and indices must have equal size along all axes except for axis {axis}')

        node.out_port(0).data.set_shape(indices_shape)

        data = node.in_port(0).data.get_value()
        indices = node.in_port(1).data.get_value()
        if data is not None and indices is not None:
            out_value = np.empty(indices_shape, dtype=data.dtype)
            for idx in np.ndindex(*indices_shape):
                data_idx = list(idx)
                data_idx[node.axis] = indices[idx]
                out_value[idx] = data[tuple(data_idx)]
            node.out_port(0).data.set_value(out_value)

        PermuteAttrs.create_permute_attrs(node, attrs=[('axis', 'input:0')])
