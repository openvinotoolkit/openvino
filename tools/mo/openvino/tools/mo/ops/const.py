# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import mo_array, int64_array
from openvino.tools.mo.middle.passes.convert_data_type import data_type_str_to_np, np_data_type_to_destination_type, \
    precision_to_destination_type
from openvino.tools.mo.ops.op import Op


class Const(Op):
    """
    Operation producing constant value stored in the attribute 'value' of shape 'shape'.
    """
    op = 'Const'

    def __init__(self, graph, attrs: dict = None):
        super().__init__(graph, {
            'type': self.op,
            'op': self.op,
            'version': 'opset1',
            'infer': self.infer,
            'value': None,
            'shape': None,
            'data_type': None,
            'out_ports_count': 1,
            'type_infer': self.type_infer,
        }, attrs)
        if not isinstance(self.attrs['value'], np.ndarray):
            self.attrs['value'] = mo_array(self.attrs['value'])

        self.attrs['shape'] = int64_array(self.attrs['value'].shape)
        if 'force_shape' in self.attrs and self.attrs['force_shape'] is not None:
            self.attrs['shape'] = int64_array(self.attrs['force_shape'])

        self.attrs['data_type'] = self.attrs['value'].dtype
        if 'force_type' in self.attrs and self.attrs['force_type'] is not None:
            self.attrs['data_type'] = data_type_str_to_np(self.attrs['force_type'])

    def supported_attrs(self):
        return [
            'offset',
            'size',
            ('shape', lambda node: ','.join([str(i) for i in node.shape])),
            ('element_type', lambda node: precision_to_destination_type(node.force_type)
            if node.has_valid('force_type') else np_data_type_to_destination_type(node.value.dtype)),
        ]

    @staticmethod
    def type_infer(node):
        node.out_port(0).set_data_type(node.value.dtype, override=True)
        if node.has_valid('force_type'):
            node.out_port(0).set_data_type(node.data_type, override=True)

    @staticmethod
    def infer(node):
        # no broadcast, copy as-is (tensor or scalar) or apply broadcast depending on value and shape
        output_value = node.value if isinstance(node.value, np.ndarray) or len(node.shape) == 0 \
            else np.full(node.shape, node.value)

        node.out_port(0).data.set_value(output_value)
