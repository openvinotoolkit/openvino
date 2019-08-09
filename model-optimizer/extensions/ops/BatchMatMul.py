"""
 Copyright (c) 2019 Intel Corporation

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

from mo.graph.graph import Graph, Node
from mo.ops.op import Op


class BatchMatMul(Op):
    op = 'BatchMatMul'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': __class__.op,
            'type': 'Gemm',
            'infer': __class__.infer,
            'in_ports_count': 2,
            'out_ports_count': 1,
        }, attrs)

    def supported_attrs(self):
        return [
            'transpose_a',
            'transpose_b',
        ]

    @staticmethod
    def infer(node: Node):
        assert (len(node.in_nodes()) == 2)

        shapes = [node.in_port(i).data.get_shape().copy() for i in range(2)]
        log.debug('BatchMatMul shapes: {}'.format(shapes))
        if any(s is None or len(s) < 2 for s in shapes):
            log.error("BatchMatMul wasn't able to infer shape")
            return

        if node.has_and_set('transpose_a'):
            shapes[0][-1], shapes[0][-2] = shapes[0][-2], shapes[0][-1]
        if node.has_and_set('transpose_b'):
            shapes[1][-1], shapes[1][-2] = shapes[1][-2], shapes[1][-1]

        log.debug('BatchMatMul shapes after transposes: {}'.format(shapes))
        if any(shapes[0][:-2] != shapes[1][:-2]) or shapes[0][-1] != shapes[1][-2]:
            log.error("MatMul wasn't able to infer shape because input dimensions are not compatible")
            return

        output_shape = np.concatenate((np.array(shapes[0][:-1], dtype=np.int64),
                                       np.array([shapes[1][-1]], dtype=np.int64)))

        node.out_port(0).data.set_shape(output_shape)
        node['channel_dims'] = output_shape.size - 1
