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

import copy

from mo.graph.graph import Node, Graph
from mo.ops.op import Op


class SpatialTransformOp(Op):
    op = 'SpatialTransformer'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': __class__.op,
            'op': __class__.op,
            'in_ports_count': 1,
            'out_ports_count': 1,
            'infer': SpatialTransformOp.sp_infer
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return [
            'transform_type',
            'sampler_type',
            'output_H',
            'output_W',
            'to_compute_dU',
            'theta_1_1',
            'theta_1_2',
            'theta_1_3',
            'theta_2_1',
            'theta_2_2',
            'theta_2_3'
        ]

    @staticmethod
    def sp_infer(node: Node):
        input_shape = node.in_node(0).shape
        output_shape = copy.copy(input_shape)
        if node.has_valid('output_H'):
            output_shape[2] = node.output_H
        if node.has_valid('output_W'):
            output_shape[3] = node.output_W
        node.out_node().shape = output_shape
