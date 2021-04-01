# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node, Graph
from mo.middle.passes.convert_data_type import np_data_type_to_destination_type
from mo.ops.op import Op


class NonZero(Op):
    op = 'NonZero'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        assert 'output_type' in attrs, 'NonZero has mandatory `output_type` attribute'
        mandatory_props = {
            'op': self.op,
            'type': self.op,
            'version': 'opset3',

            'infer': self.infer,
            'type_infer': self.type_infer,

            'in_ports_count': 1,
            'out_ports_count': 1,
        }
        super().__init__(graph, mandatory_props, attrs)

    def backend_attrs(self):
        return [
            ('output_type', lambda node: np_data_type_to_destination_type(node.output_type)),
        ]

    @staticmethod
    def infer(node: Node):
        node_name = node.soft_get('name', node.id)
        input_shape = node.in_port(0).data.get_shape()
        assert input_shape is not None, 'The input shape for node "{}" is None'.format(node_name)
        assert node.has_valid('output_type'), \
            '`output_type` attribute is not set for NonZero node `{}`'.format(node_name)
        assert node.output_type in [np.int64, np.int32], \
            'NonZero `output_type` attribute must be int32 or int64, `{}` found'.format(np.dtype(node.output_type).name)

        input_value = node.in_port(0).data.get_value()
        if input_value is not None:
            node.out_port(0).data.set_value(np.array(np.nonzero(input_value), dtype=node.output_type))
        else:
            # output shape of NonZero should be [input_rank, dynamic]
            # having restriction to save IR with static shape only we count upper-bound shape value here
            node.out_port(0).data.set_shape(int64_array([len(input_shape), np.prod(input_shape)]))

    @staticmethod
    def type_infer(node):
        assert node.output_type in [np.int64, np.int32], \
            'NonZero `output_type` attribute must be int32 or int64, `{}` found'.format(np.dtype(node.output_type).name)
        node.out_port(0).set_data_type(node.output_type)
