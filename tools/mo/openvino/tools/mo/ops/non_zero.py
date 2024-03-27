# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import is_fully_defined, dynamic_dimension_value
from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.middle.passes.convert_data_type import np_data_type_to_destination_type
from openvino.tools.mo.ops.op import Op


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
        if is_fully_defined(input_value):
            node.out_port(0).data.set_value(mo_array(np.nonzero(input_value), dtype=node.output_type))
        else:
            if is_fully_defined(input_shape):
                # output shape of NonZero is still static (upper bound)
                node.out_port(0).data.set_shape([len(input_shape), np.prod(input_shape)])
            else:
                node.out_port(0).data.set_shape([len(input_shape), dynamic_dimension_value])

    @staticmethod
    def type_infer(node):
        assert node.output_type in [np.int64, np.int32], \
            'NonZero `output_type` attribute must be int32 or int64, `{}` found'.format(np.dtype(node.output_type).name)
        node.out_port(0).set_data_type(node.output_type)
