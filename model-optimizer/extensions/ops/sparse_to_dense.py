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
from mo.ops.op import Op


class SparseToDense(Op):
    """ The operation converts a sparse tensor to a dense tensor.
        For more details see https://www.tensorflow.org/api_docs/python/tf/sparse/to_dense

        4 inputs:
            - [0, required] input indices of the sparse tensor (2D),
            - [1, required] shape of the sparse tensor. Value of this input is required for the Model Optimizer (1D),
            - [2, required] input values of the sparse tensor (1D),
            - [3, optional] default value to insert at missing positions (0D). If it is not specified, zero value is used.
        
        output:
            - dense tensor (2D)
    """

    op = 'SparseToDense'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': __class__.op,
            'op': __class__.op,
            'version': 'experimental',
            'type_infer': self.type_infer,
            'infer': self.infer,
            'in_ports_count': 4,
            'out_ports_count': 1,
        }, attrs)

    def supported_attrs(self):
        return []

    @staticmethod
    def type_infer(node):
        # output data type must be the same as input values type
        values_type = node.in_port(1).get_data_type()
        node.out_port(0).set_data_type(values_type)

    @staticmethod
    def infer(node: Node):
        assert len(node.in_nodes()) == 4 or len(node.in_nodes()) == 3, \
            "Incorrect number of inputs for {} node".format(node.id)

        dense_shape = node.in_port(1).data.get_value()

        # check that shape value is defined that is needed for shape inference
        assert dense_shape is not None, \
            "SparseToDense is supported only with constant shape value"

        node.out_port(0).data.set_shape(np.array(dense_shape, dtype=np.int64))

        input_indices = node.in_port(0).data.get_value()
        input_values = node.in_port(2).data.get_value()
        default_value = np.float32(0.0)
        if not node.in_port(3).disconnected():
            default_value = node.in_port(3).data.get_value()

        # compute the output value if all input is constant
        if input_indices is not None and input_values is not None and default_value is not None:
            output_value = np.full(dense_shape, default_value)
            for input_index, input_value in zip(input_indices, input_values):
                output_value[tuple(input_index)] = input_value
            node.out_port(0).data.set_value(output_value)
