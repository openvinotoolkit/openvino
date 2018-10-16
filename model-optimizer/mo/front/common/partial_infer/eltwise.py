"""
 Copyright (c) 2018 Intel Corporation

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
import logging as log

from mo.graph.graph import get_sorted_inputs, Node


def eltwise_infer(node, op=None, **kwargs):
    inputs = [Node(node.graph, inp) for inp, attr in get_sorted_inputs(node)
              if 'control_flow_edge' not in attr or not attr['control_flow_edge']]
    shapes = [node.graph.node[inp.id]['shape'] for inp in inputs]
    values = [node.graph.node[inp.id]['value'] for inp in inputs]

    # infer output shape based on input shapes without op involvement
    # based on repeated application of rules https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

    if any([s is None for s in shapes]):
        # nothing is known
        return

    if node.has_valid('axis') and all([value is None for value in values]):
        log.error('Eltwise operation with axis is not supported')
        return

    def check_value(value):
        # Check that value has shape like N,1,1
        return np.prod(value.shape) == np.max(value.shape) and \
                       value.shape[0] == np.max(value.shape)

    # make all input shapes of the same size by adding leading 1's
    max_dims = max([len(s) for s in shapes])
    # In case of not None axis, we are extending shape for not None values according to axis
    if node.has_valid('axis'):
        # Check if axis match feature dim and values shapes suits so that is ok, else we mark this op with can_be_fused=False
        if node.axis == node.graph.graph['feature_dim'] and \
           all([check_value(value) for value in values if value is not None]):
            for id, value in enumerate(values):
                if value is not None:
                    # Expand dims for value
                    dims_to_add = max_dims - node.axis - len(value.shape) # how much 1 we should add to the shape
                    if dims_to_add < 0:
                        log.error('Axis attribute for {} node is wrong (axis={}, input_shapes={})'.format(node.name, node.axis, shapes))
                        return
                    # Update values and shapes with new shape
                    shape = np.append(value.shape, [1]*dims_to_add).astype(dtype=np.int64)
                    value = np.reshape(value, shape)
                    shapes[id], values[id] = np.array(shape), np.array(value)
                    # Update node weights & shape
                    inputs[id].value, inputs[id].shape = np.array(value), np.array(shape)
        else:
            node['can_be_fused'] = False


    extended_shapes = [np.concatenate((np.ones(max_dims - len(s), dtype=np.int64), s)) for s in shapes]
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

    node.out_node().value = op(*values, **kwargs)
