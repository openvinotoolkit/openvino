"""
 Copyright (c) 2018-2019 Intel Corporation

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

import networkx as nx
import numpy as np

from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node


def eltwise_infer(node, op=None, **kwargs):
    raw_inputs = [(inp, attr) for inp, attr in node.get_sorted_inputs()
                  if 'control_flow_edge' not in attr or not attr['control_flow_edge']]
    inputs = [Node(node.graph, inp) for inp, attr in raw_inputs]
    shapes = [node.graph.node[inp]['shape'] for inp, attr in raw_inputs]
    values = [node.graph.node[inp]['value'] for inp, attr in raw_inputs]

    # infer output shape based on input shapes without op involvement
    # based on repeated application of rules https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

    if any([s is None for s in shapes]):
        # nothing is known
        return

    max_dims = None
    for id, s in enumerate(shapes):
        if max_dims is None or len(s) > max_dims:
            max_dims = len(s)

    # Make all input shapes of the same size by adding 1's
    axis = node.axis if node.has_valid('axis') else None
    for id, item in enumerate(zip(shapes, values)):
        shape, value = item
        if len(shape) != max_dims and len(shape) > 0 and axis is not None:
            new_shape = shape

            # Extend shape with 1's
            for cnt in range(axis + len(shape), max_dims):
                new_shape = np.append(new_shape, 1)

            shapes[id] = new_shape

            # Save shape for further transformation that applies this shapes for input nodes
            # We set new_shape attribute on edge for given input node
            edge_attrs = node.graph.get_edge_data(inputs[id].id, node.id)[0]

            nx.set_edge_attributes(G=node.graph,
                                   values={(inputs[id].id, node.id, 0): new_shape},
                                   name='new_shape')

            # Reshape value to correctly calculate output shape
            if values[id] is not None:
                values[id] = np.reshape(values[id], new_shape)

    extended_shapes = int64_array([np.concatenate((np.ones(max_dims - len(s), dtype=np.int64), s)) for s in shapes])
    # ugly but clear solution
    output_shape = extended_shapes[0]
    for si in range(1, len(extended_shapes)):
        for ei in range(max_dims):
            mind = min(output_shape[ei], extended_shapes[si][ei])
            maxd = max(output_shape[ei], extended_shapes[si][ei])
            if mind == -1:
                output_shape[ei] = -1
            elif mind == 1:
                output_shape[ei] = maxd
            elif mind != maxd:
                output_shape[ei] = -1
    node.out_node().shape = output_shape

    if op is None or any([v is None for v in values]):
        return

    if len(values) <= 2:
        node.out_node().value = op(*values, **kwargs)
    else:
        node.out_node().value = values[0]
        for i in range(len(values) - 1):
            node.out_node().value = op(node.out_node().value, values[i + 1])


def bias_add_infer(node, op):
    if node.in_port(0).data.get_value() is not None and node.in_port(1).data.get_value() is not None and op is not None:
        node.out_port(0).data.set_value(op(node.in_port(0).data.get_value(), node.in_port(1).data.get_value()))
    else:
        node.out_port(0).data.set_shape(node.in_port(0).data.get_shape())
