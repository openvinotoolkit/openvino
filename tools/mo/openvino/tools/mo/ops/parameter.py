# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import unmask_shape
from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.middle.passes.convert_data_type import np_data_type_to_destination_type
from openvino.tools.mo.ops.op import Op, PermuteAttrs


class Parameter(Op):
    op = 'Parameter'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'op': self.op,
            'type': self.op,
            'version': 'opset1',

            'infer': self.infer,
            'reverse_infer': self.reverse_infer,
            'is_input': True,
            'data_type': None,

            'type_infer': self.type_infer,

            'out_ports_count': 1,
        }
        if 'data_type' not in attrs:
            mandatory_props['data_type'] = np.float32
        super().__init__(graph, mandatory_props, attrs)

    @staticmethod
    def type_infer(node):
        node.out_port(0).set_data_type(node.data_type)

    def supported_attrs(self):
        return [
            ('shape', lambda node: ','.join([str(i) for i in unmask_shape(node.shape)])),
            ('element_type', lambda node: np_data_type_to_destination_type(node.data_type)),
        ]

    @staticmethod
    def infer(node):
        name = node.soft_get('name', node.id)
        assert node.has_valid('shape'), \
            'Parameter node {} should have `shape` attribute. Please use cli options to set model input shape' \
            ''.format(name)
        node.out_port(0).data.set_shape(node.shape)

        PermuteAttrs.create_permute_attrs(node, attrs=[('shape', 'output:0')])

    @staticmethod
    def reverse_infer(node: Node):
        # update node 'shape' attribute (if it is not defined) from the output port shape which was calculated
        # during the reverse_infer phase
        shape = node.soft_get('shape', None)
        if shape is not None:
            return
        for i, out_port in node.out_ports().items():
            out_shape = out_port.data.get_shape()
            if out_shape is not None:
                node['shape'] = out_shape
