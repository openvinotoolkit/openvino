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
from mo.middle.passes.convert_data_type import np_data_type_to_destination_type
from mo.ops.op import Op


class Bucketize(Op):
    op = 'Bucketize'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'kind': 'op',
            'type': __class__.op,
            'op': __class__.op,
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
            return ['with_right_bound']
        else:
            return [
                'with_right_bound',
                ('output_type', lambda node: np_data_type_to_destination_type(node.output_type)),
            ]

    @staticmethod
    def type_infer(node):
        # the output is always integer since the layer outputs a bucket index
        if node.get_opset() == "extension":
            node.out_port(0).set_data_type(np.int32)
        else:
            assert node.output_type in [np.int64, np.int32], \
                'Bucketize `output_type` attribute must be int32 or int64, `{}` found'.format(np.dtype(node.output_type).name)
            node.out_port(0).set_data_type(node.output_type)

    @staticmethod
    def infer(node: Node):
        node_name = node.soft_get('name', node.id)
        assert node.with_right_bound is not None, \
            "Attribute \"with_right_bound\" is not defined"
        assert len(node.in_nodes()) == 2, \
            "Incorrect number of inputs for {} node".format(node.id)
        if node.get_opset() == "extension":
            output_type = np.int32
        else:
            assert node.has_valid('output_type'), \
                '`output_type` attribute is not set for Bucketize node `{}`'.format(node_name)
            assert node.output_type in [np.int64, np.int32], \
                'Bucketize `output_type` attribute must be int32 or int64, `{}` found'.format(np.dtype(node.output_type).name)
            output_type = node.output_type

        output_shape = node.in_port(0).data.get_shape()
        node.out_port(0).data.set_shape(output_shape)

        input_value = node.in_port(0).data.get_value()
        buckets_value = node.in_port(1).data.get_value()

        # compute if all input is constant
        if input_value is not None and buckets_value is not None:
            node.out_port(0).data.set_value(np.array(np.digitize(input_value, buckets_value, right=node.with_right_bound), dtype=node.output_type))
