"""
 Copyright (c) 2019 Intel Corporation

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
from mo.graph.perm_inputs import PermuteInputs
from mo.ops.op import Op

reduce_map = {
    'ReduceSum': np.sum,
    'ReduceProd': np.prod,
    'ReduceMax': np.max,
    'ReduceMin': np.min,
    'ReduceMean': np.mean,
    'ReduceAnd': np.all,
}


def reduce_infer(node: Node):
    connected_in_ports = [port for port in node.in_ports().values() if not port.disconnected()]
    assert len(connected_in_ports) == 2, \
        "{} node `{}` should have 2 input ports, where 0-input is data input and 1-input represent " \
        "`reduction_indices`".format(node.op, node.id)

    in_data = node.in_port(0).data
    in_shape = in_data.get_shape()
    axis = node.in_port(1).data.get_value()

    # If the axis is None then reduce over all the dimensions of the input tensor
    if axis.size == 1 and axis.item() is None:
        axis = int64_array(list(range(len(in_shape))))
        node.in_port(1).data.set_value(axis)

    assert in_shape is not None, "Can not infer {} node `{}`: shape of 0-input unknown".format(node.op, node.id)

    axis = axis.copy()
    if axis.size == 1:
        axis = int64_array([axis.item()])

    in_value = in_data.get_value()

    if in_value is not None:
        value = reduce_map[node.op](in_value.copy(), axis=tuple(axis), keepdims=node.keep_dims)
        node.out_port(0).data.set_value(value)
    else:
        used_dims = np.zeros(len(in_shape), dtype=np.bool)
        output_shape = in_shape.copy()

        for dim in axis:
            used_dims[dim] = True
            output_shape[dim] = 1

        # In case if keep dims == False, we should remove all 1 dims that was used in reduction
        if not node.keep_dims:
            output_shape = output_shape[np.invert(used_dims)]

        node.out_port(0).data.set_shape(output_shape)

    PermuteInputs().set_input_permutation(node.in_node(1), node, 'input:0', 'axis')


class ReduceOp(Op):
    enabled = False
    op = None

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': self.op,
            'type': self.op,
            'infer': reduce_infer,
            'in_ports_count': 2,
            'out_ports_count': 1,
            'force_precision_in_ports': {1: 'int32'},
        }, attrs)

    def supported_attrs(self):
        return [
            'keep_dims',
        ]


class ReduceSum(ReduceOp):
    enabled = True
    op = 'ReduceSum'


class ReduceProd(ReduceOp):
    op = 'ReduceProd'
    enabled = True


class ReduceMin(ReduceOp):
    op = 'ReduceMin'
    enabled = True


class ReduceMax(ReduceOp):
    op = 'ReduceMax'
    enabled = True


class ReduceMean(ReduceOp):
    op = 'ReduceMean'
    enabled = True


class ReduceAnd(ReduceOp):
    op = 'ReduceAnd'
    enabled = True
