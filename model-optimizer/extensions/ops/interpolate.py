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

from mo.graph.graph import Node, Graph
from mo.ops.op import Op, PermuteAttrs


class Interpolate(Op):
    op = 'Interpolate'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'op': self.op,
            'type': self.op,
            'version': 'opset1',

            'axes': None,
            'mode': None,
            'align_corners': 0,
            'antialias': 0,
            'pads_begin': 0,
            'pads_end': 0,

            'infer': self.infer,

            'force_precision_in_ports': {1: 'int64'},

            'in_ports_count': 2,
            'out_ports_count': 1,
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        attributes_for_opsets = {
            'opset1': [
                ('axes', lambda node: ','.join(map(str, node.axes))),
                'mode', 'align_corners', 'antialias', 'pads_begin', 'pads_end',
            ],
            'opset3': [
                ('axes', lambda node: ','.join(map(str, node.axes))),
                'mode', 'antialias', 'nearest_mode', 'cube_coeff', 'coordinate_transformation_mode',
                ('pads_begin', lambda node: pad_attribute_to_str(node, 'pads_begin')),
                ('pads_end', lambda node: pad_attribute_to_str(node, 'pads_end')),
            ]
        }

        opset = self.get_opset()
        if opset in attributes_for_opsets:
            attributes = attributes_for_opsets[opset]
        else:
            attributes = attributes_for_opsets['opset1']

        return attributes

    @staticmethod
    def infer(node: Node):
        assert len([p for p in node.in_ports().values() if not p.disconnected()]) == 2
        assert node.has_valid('mode')
        assert node.has_valid('axes')

        infers = {
            'opset1': Interpolate.infer_for_opset1,
            'opset3': Interpolate.infer_for_opset3,
        }
        if node.has_valid('version') and node.version in infers:
            infer_func = infers[node.version]
        else:
            infer_func = infers['opset1']

        infer_func(node)

    @staticmethod
    def infer_for_opset1(node: Node):
        src_shape = node.in_port(0).data.get_shape()

        assert src_shape is not None
        dst_shape = node.in_port(1).data.get_value()
        assert dst_shape is not None

        output_shape = src_shape.copy()
        for ind, axis in enumerate(node.axes):
            output_shape[axis] = dst_shape[ind]

        node.out_port(0).data.set_shape(output_shape)

        PermuteAttrs.create_permute_attrs(node, attrs=[('axes', 'input:0')])

    @staticmethod
    def infer_for_opset3(node: Node):
        src_shape = node.in_port(0).data.get_shape()
        assert src_shape is not None

        input_rank = len(src_shape)

        pads_begin = correct_pad(node.soft_get('pads_begin', [0]), input_rank)
        pads_end = correct_pad(node.soft_get('pads_end', [0]), input_rank)
        node['pads_begin'] = pads_begin
        node['pads_end'] = pads_end

        axes = node.axes
        dst_shape = node.in_port(1).data.get_value()
        assert dst_shape is not None

        output_shape = src_shape + pads_begin + pads_end
        for i in range(0, len(axes)):
            output_shape[axes[i]] = dst_shape[i]

        node.out_port(0).data.set_shape(output_shape)

        PermuteAttrs.create_permute_attrs(node, attrs=[('axes', 'input:0')])


def pad_attribute_to_str(node: Node, attr: str):
    return ','.join(map(str, node[attr])) if node.has_valid(attr) else None


def correct_pad(pad, rank):
    pad_len = len(pad)
    if pad_len < rank:
        return np.pad(pad, (0, rank - pad_len), 'constant').astype(np.int64)
    elif pad_len > rank:
        return np.array(pad[: rank]).astype(np.int64)
    else:
        return np.array(pad, dtype=np.int64)

