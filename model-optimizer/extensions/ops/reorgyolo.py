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

import networkx as nx
import numpy as np

from mo.graph.graph import Node, Graph
from mo.ops.op import Op, PermuteAttrs


class ReorgYoloOp(Op):
    op = 'ReorgYolo'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': __class__.op,
            'op': __class__.op,
            'infer': ReorgYoloOp.reorgyolo_infer
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return [
            'stride'
        ]

    @staticmethod
    def reorgyolo_infer(node: Node):
        input_shape = node.in_node(0).shape
        if input_shape is None:
            return

        stride = node.stride

        output_shape = np.full_like(input_shape, -1, dtype=np.int64)
        output_shape[node.batch_dims] = input_shape[node.batch_dims]  # pylint: disable=unsupported-assignment-operation
        output_shape[node.channel_dims] = input_shape[node.channel_dims] * stride ** 2  # pylint: disable=unsupported-assignment-operation
        # Round as in caffe
        output_shape[node.spatial_dims] = np.round(input_shape[node.spatial_dims] / stride)  # pylint: disable=unsupported-assignment-operation

        node.out_node().shape = output_shape
        PermuteAttrs.create_permute_attrs(node, attrs=[('channel_dims', 'input:0'), ('spatial_dims', 'input:0')])
