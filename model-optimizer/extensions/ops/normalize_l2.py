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
import numpy as np

from mo.graph.graph import Graph, Node
from mo.ops.op import Op


class NormalizeL2Op(Op):
    op = 'NormalizeL2'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'kind': 'op',
            'type': __class__.op,
            'op': __class__.op,
            'eps': None,
            'p': None,
            'eps_mode': None,
            'in_ports_count': 2,
            'out_ports_count': 1,
            'infer': __class__.infer
        }, attrs)

    def supported_attrs(self):
        return ['eps', 'eps_mode']

    @staticmethod
    def infer(node: Node):
        input_shape = node.in_port(0).data.get_shape()
        if input_shape is None:
            return

        input_value = node.in_port(0).data.get_value()
        axes = node.in_port(1).data.get_value()
        if input_value is not None and axes is not None:
            norm_value = np.linalg.norm(input_value, node.p, axes, keepdims=True)
            if node.eps_mode == 'add':
                norm_value = norm_value + node.eps
            elif node.eps_mode == 'max':
                norm_value = np.max(norm_value, node.eps)
            else:
                assert False, 'Unsupported "eps_mode" = {}'.format(node.eps_mode)
            node.out_port(0).data.set_value(input_value / norm_value)
        else:
            node.out_port(0).data.set_shape(input_shape)
