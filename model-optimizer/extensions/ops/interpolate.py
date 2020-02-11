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

from mo.front.common.layout import get_batch_dim, get_features_dim, shape_for_layout
from mo.graph.graph import Node, Graph
from mo.ops.op import Op, PermuteAttrs


class Interpolate(Op):
    op = 'Interpolate'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': __class__.op,
            'op': __class__.op,
            'axes': None,
            'mode': None,
            'align_corners': 0,
            'antialias': 0,
            'pads_begin': 0,
            'pads_end': 0,
            'in_ports_count': 2,
            'out_ports_count': 1,
            'force_precision_in_ports': {1: 'int64'},
            'infer': __class__.infer,
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return [
            ('axes', lambda node: ','.join(map(str, node.axes))),
            'mode', 'align_corners', 'antialias', 'pads_begin', 'pads_end',
        ]

    @staticmethod
    def infer(node: Node):
        assert len([p for p in node.in_ports().values() if not p.disconnected()])
        assert node.has_valid('mode')
        assert node.has_valid('axes')

        src_shape = node.in_port(0).data.get_shape()

        assert src_shape is not None
        dst_shape = node.in_port(1).data.get_value()
        assert dst_shape is not None

        output_shape = src_shape.copy()
        for ind, axis in enumerate(node.axes):
            output_shape[axis] = dst_shape[ind]

        node.out_node().shape = output_shape

        PermuteAttrs.create_permute_attrs(node, attrs=[('axes', 'input:0')])

