# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import is_fully_defined, shape_array, dynamic_dimension_value
from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.op import Op


class Size(Op):
    op = 'Size'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        assert 'output_type' in attrs, 'Size has mandatory `output_type` attribute'

        mandatory_props = {
            'type': None,
            'op': self.op,

            'output_type': np.int64,
            'infer': self.infer,
        }
        super().__init__(graph, mandatory_props, attrs)

    @staticmethod
    def infer(node: Node):
        name = node.soft_get('name', node.id)
        connected_in_ports = [port for port in node.in_ports().values() if not port.disconnected()]
        assert len(connected_in_ports) == 1, \
            'Size operation should have exact one input node, but it has {}'.format(len(connected_in_ports))

        input_shape = node.in_port(0).data.get_shape()
        assert input_shape is not None, \
            'Input shape is undefined for Size node `{}`'.format(node.soft_get('name', node.id))

        assert node.has_valid('output_type'), \
            '`output_type` attribute is not set for Size node `{}`'.format(name)
        assert node.output_type in [np.int64, np.int32], \
            'Size `output_type` attribute must be int32 or int64, `{}` found'.format(np.dtype(node.output_type).name)

        if is_fully_defined(input_shape):
            node.out_port(0).data.set_value(mo_array(np.prod(input_shape), dtype=node.output_type))
        else:
            node.out_port(0).data.set_value(shape_array(dynamic_dimension_value))
