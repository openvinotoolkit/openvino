# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from mo.graph.graph import Graph
from mo.middle.passes.convert_data_type import np_data_type_to_destination_type
from mo.ops.op import Op, PermuteAttrs


class Parameter(Op):
    op = 'Parameter'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'op': self.op,
            'type': self.op,
            'version': 'opset1',

            'infer': self.infer,
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
            ('shape', lambda node: ','.join([str(i) for i in node.shape])),
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
