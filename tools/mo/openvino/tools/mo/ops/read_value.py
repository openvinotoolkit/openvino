# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import numpy as np

from openvino.tools.mo.ops.op import Op
from openvino.tools.mo.middle.passes.convert_data_type import np_data_type_to_destination_type
from openvino.tools.mo.front.common.partial_infer.utils import unmask_shape
from openvino.tools.mo.graph.graph import Graph, Node
from openvino.runtime import PartialShape

class ReadValue(Op):
    op = 'ReadValue'

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': self.op,
            'op': self.op,
            'version': 'opset6',
            'infer': self.infer,
            'type_infer': self.type_infer,
            'in_ports_count': 1,
            'out_ports_count': 1,
        }, attrs)

    @staticmethod
    def shape_serialize(node):
        shape = node.soft_get('variable_shape')
        if isinstance(shape, np.ndarray):
            return shape.tolist()
        if isinstance(shape, np.ma.masked_array):
            shape = unmask_shape(shape)
        if isinstance(shape, PartialShape):
            return shape.to_string()
        raise Exception("Unknown shape type in user_shape attribute {}".format(type(shape)))

    def backend_attrs(self):
        return ['variable_id',
                ('variable_shape', lambda node: self.shape_serialize(node)),
                ('variable_type', lambda node: np_data_type_to_destination_type(node.variable_type))]

    @staticmethod
    def type_infer(node: Node):
        node.variable_type = node.in_port(0).get_data_type()
        node.out_port(0).set_data_type(node.in_port(0).get_data_type())

    @staticmethod
    def infer(node: Node):
        assert node.has_valid('variable_id'), \
            "There is no required attribute variable_id in ReadValue op with name " + node.id
        in_shape = node.in_port(0).data.get_shape()
        node.out_port(0).data.set_shape(in_shape)
        node.variable_shape = in_shape
