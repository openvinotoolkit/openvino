# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.front.extractor import bool_to_str
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.middle.passes.convert_data_type import np_data_type_to_destination_type
from openvino.tools.mo.ops.op import Op


class Bucketize(Op):
    op = 'Bucketize'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'kind': 'op',
            'type': self.op,
            'op': self.op,
            'version': 'opset3',

            'type_infer': self.type_infer,
            'infer': self.infer,

            'in_ports_count': 2,
            'out_ports_count': 1,
        }
        super().__init__(graph, mandatory_props, attrs)

    def backend_attrs(self):
        version = self.get_opset()
        if version == "extension":
            return [('with_right_bound', lambda node: bool_to_str(node, 'with_right_bound'))]
        else:
            return [
                ('with_right_bound', lambda node: bool_to_str(node, 'with_right_bound')),
                ('output_type', lambda node: np_data_type_to_destination_type(node.output_type)),
            ]

    @staticmethod
    def type_infer(node):
        # the output is always integer since the layer outputs a bucket index
        if node.get_opset() == "extension":
            node.out_port(0).set_data_type(np.int32)
        else:
            assert node.output_type in [np.int64, np.int32], \
                'Bucketize `output_type` attribute must be int32 or int64, `{}` found' \
                ''.format(np.dtype(node.output_type).name)
            node.out_port(0).set_data_type(node.output_type)

    @staticmethod
    def infer(node: Node):
        node_name = node.soft_get('name', node.id)
        assert node.with_right_bound is not None, \
            "Attribute \"with_right_bound\" is not defined"
        assert len(node.in_nodes()) == 2, \
            "Incorrect number of inputs for {} node".format(node.id)
        if node.get_opset() != "extension":
            assert node.has_valid('output_type'), \
                '`output_type` attribute is not set for Bucketize node `{}`'.format(node_name)
            assert node.output_type in [np.int64, np.int32], \
                'Bucketize `output_type` attribute must be int32 or int64, `{}` found'.format(np.dtype(node.output_type).name)

        output_shape = node.in_port(0).data.get_shape()
        node.out_port(0).data.set_shape(output_shape)

        input_value = node.in_port(0).data.get_value()
        buckets_value = node.in_port(1).data.get_value()

        # compute if all input is constant
        if input_value is not None and buckets_value is not None:
            node.out_port(0).data.set_value(mo_array(np.digitize(input_value, buckets_value, right=node.with_right_bound), dtype=node.output_type))
