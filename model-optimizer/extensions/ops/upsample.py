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

import math
import numpy as np

from mo.front.common.layout import get_batch_dim, get_features_dim, get_height_dim, get_width_dim, shape_for_layout
from mo.graph.graph import Node, Graph
from mo.ops.op import Op


class UpsampleOp(Op):
    op = 'Upsample'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'op': __class__.op,
            'in_ports_count': 2,
            'out_ports_count': 1,
            'infer': UpsampleOp.upsample_infer
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return [
            'height_scale',
            'width_scale',
            'mode',
        ]

    @staticmethod
    def upsample_infer(node: Node):
        layout = node.graph.graph['layout']
        assert len(layout) == 4

        input_shape = node.in_node(0).shape
        if input_shape is None:
            return

        if len(node.in_nodes()) == 1:
            in_height = input_shape[get_height_dim(layout, 4)]
            in_width = input_shape[get_width_dim(layout, 4)]
            assert node.has('width_scale') is not None and node.has('height_scale') is not None
            out_height = math.floor(in_height * node.height_scale)
            out_width = math.floor(in_width * node.width_scale)
            node.out_node().shape = shape_for_layout(layout,
                                                     batch=input_shape[get_batch_dim(layout, 4)],
                                                     features=input_shape[get_features_dim(layout, 4)],
                                                     height=out_height,
                                                     width=out_width)
        else:
            assert node.in_node(1).value is not None
            eps = 1e-5  # This is to make rounding in case of very close number to round to closest instead of down
            # generic output shape calculation to support 5D input shape case
            node.out_node().shape = np.array((input_shape + eps) * node.in_node(1).value).astype(np.int64)
