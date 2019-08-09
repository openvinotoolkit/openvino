"""
 Copyright (c) 2018-2019 Intel Corporation

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

from mo.front.common.partial_infer.elemental import copy_shape_infer
from mo.graph.graph import Graph
from mo.ops.op import Op
from mo.front.common.partial_infer.utils import mark_input_bins


class PreluOp(Op):
    op = 'PReLU'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': __class__.op,
            'op': __class__.op,
            'in_ports_count': 1,
            'out_ports_count': 1,
            'infer': PreluOp.prelu_shape_infer
        }, attrs)

    def supported_attrs(self):
        return ['channel_shared', 'filler_type', 'filler_value', 'min', 'max', 'mean', 'std', 'sparse', 'variance_norm']

    @staticmethod
    def prelu_shape_infer(node):
        if len(node.in_nodes()) == 2:
            gamma_vector = node.in_node(1)
            if np.all(gamma_vector.shape == [1]):
                node['channel_shared'] = 1
            else:
                node['channel_shared'] = 0
            mark_input_bins(node)
        copy_shape_infer(node)
