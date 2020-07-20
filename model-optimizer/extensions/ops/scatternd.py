"""
 Copyright (C) 2020 Intel Corporation

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

from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node, Graph
from mo.ops.op import Op


class ScatterNDBase(Op):
    enabled = False

    op = op_type = None
    version = None

    def __init__(self, graph: Graph, attrs: dict):
        assert self.op is not None and self.op_type is not None and self.version is not None, \
            'Please use specialized ScatterNDBase operation class, ScatterNDBase is base class'

        mandatory_props = {
            'op': self.op,
            'type': self.op_type,
            'version': self.version,

            'infer': self.infer,

            'in_ports_count': 3,
            'out_ports_count': 1,
        }
        super().__init__(graph, mandatory_props, attrs)

    @staticmethod
    def infer(node: Node):
        node_name = node.soft_get('name', node.id)

        input_shape = node.in_port(0).data.get_shape()
        indices_shape = node.in_port(1).data.get_shape()
        updates_shape = node.in_port(2).data.get_shape()
        assert input_shape is not None and updates_shape is not None and indices_shape is not None, \
            'The node "{}" input shape is None'.format(node_name)

        # check that shapes are correct
        # 1. ranks of both input and indices must be at least 1
        assert len(input_shape) >= 1 and len(indices_shape) >= 1, \
            'The node "{}" input and indices ranks must be at least 1'.format(node_name)
           
        # 2. the last dimension of indices shape must be at most a rank of input
        assert indices_shape[-1] <= len(input_shape), \
            'The last dimension of indices shape must be at most a rank of input for the node "{}"'.format(node_name)

        # 3. updates is a tensor of shape indices_shape[:-1] + input_shape[indices_shape[-1]:]
        expected_updates_shape = np.concatenate((indices_shape[:-1], input_shape[indices_shape[-1]:]), axis=0)
        assert np.array_equal(updates_shape, expected_updates_shape), \
            'The updates shape must be equal to indices_shape[:-1] + input_shape[indices_shape[-1]:] for the node "{}"'.format(node_name)

        node.out_port(0).data.set_shape(input_shape)

    @staticmethod
    def type_infer(node: Node):
        assert node.in_port(0).get_source().get_data_type() == node.in_port(2).get_source().get_data_type(), \
            'The data type of the first and the third inputs must be equal for the node {}'.format(node.name)
        node.out_port(0).set_data_type(node.in_port(0).get_data_type())


class ScatterNDUpdate(ScatterNDBase):
    op = op_type = 'ScatterNDUpdate'
    version = 'opset4'

    @staticmethod
    def infer(node: Node):
        ScatterNDBase.infer(node)

        input_value = node.in_port(0).data.get_value()
        indices_shape = node.in_port(1).data.get_shape()
        indices_value = node.in_port(1).data.get_value()
        updates_value = node.in_port(2).data.get_value()

        # compute output value if all inputs are constant
        if input_value is not None and indices_value is not None and updates_value is not None:
            output_value = input_value.copy()
            indx_range = int64_array(indices_shape[:-1])
            for indx in np.ndindex(tuple(indx_range)):
                if indx == ():
                    # a case when updates is a scalar
                    indx = 0
                    updates_value = [updates_value]
                output_value[indices_value[indx]] = updates_value[indx]

            node.out_port(0).data.set_value(output_value)
