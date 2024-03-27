# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import dynamic_dimension_value, shape_array
from openvino.tools.mo.front.extractor import bool_to_str
from openvino.tools.mo.graph.graph import Graph, Node

from openvino.tools.mo.ops.op import Op


class Multinomial(Op):
    op = 'Multinomial'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': self.op,
            'op': self.op,
            'version': 'opset13',
            'infer': self.infer,
            'in_ports_count': 2,
            'out_ports_count': 1,
            'type_infer': self.type_infer,
            'with_replacement': False,
            'log_probs': False,
            'global_seed': 0,
            'op_seed': 0,
            'convert_type': np.int64,
        }, attrs)

    def backend_attrs(self):
        return ['convert_type',
                ('with_replacement', lambda node: bool_to_str(
                    node, 'with_replacement')),
                ('log_probs', lambda node: bool_to_str(node, 'log_probs')),
                'global_seed',
                'op_seed']

    def supported_attrs(self):
        return ['convert_type',
                'with_replacement',
                'log_probs',
                'global_seed',
                'op_seed']

    @staticmethod
    def type_infer(node: Node):
        assert node.has_valid('convert_type')
        if node['convert_type'] == 'i32':
            node.out_port(0).set_data_type(np.int32)
        else:
            node.out_port(0).set_data_type(np.int64)

    @staticmethod
    def infer(node: Node):

        input_shape = node.in_node(0).shape
        output_shape = []
        if input_shape is not None and input_shape.size == 2:
            output_shape.append(input_shape[0])

        num_samples = node.in_port(1).data.get_value()
        if num_samples is not None:
            output_shape.append(np.array(num_samples).item())
        else:
            output_shape.append(dynamic_dimension_value)
        node.out_port(0).data.set_shape(shape_array(output_shape))
