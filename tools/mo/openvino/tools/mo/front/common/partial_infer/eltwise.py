# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import dynamic_dimension, dynamic_dimension_value
from openvino.tools.mo.utils.error import Error


def eltwise_infer(node, op=None, **kwargs):
    def broadcast_dims(dim1, dim2):
        if dim1 is not dynamic_dimension and dim2 is not dynamic_dimension:
            mind = min(dim1, dim2)
            maxd = max(dim1, dim2)
            if mind == 1:
                return maxd
            elif mind != maxd:
                raise Error('Input shapes mismatch for node {}: {}'.format(node_name, shapes))
            return mind
        elif dim1 is dynamic_dimension and dim2 is dynamic_dimension:
            return dynamic_dimension_value
        elif dim1 is dynamic_dimension and dim2 is not dynamic_dimension:
            return broadcast_dims(dim2, dim1)
        else:  # dim1 is static, dim2 is dynamic
            if dim1 != 1:
                return dim1
            else:
                return dim2

    raw_inputs = [(inp, attr) for inp, attr in node.get_sorted_inputs()
                  if 'control_flow_edge' not in attr or not attr['control_flow_edge']]
    shapes = [node.graph.node[inp]['shape'] for inp, attr in raw_inputs]
    values = [node.graph.node[inp]['value'] for inp, attr in raw_inputs]
    node_name = node.soft_get('name', node.id)

    if any([s is None for s in shapes]):
        raise Error('One of the input shapes for node "{}" is None'.format(node_name))

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
                new_shape = np.ma.append(new_shape, 1)

            shapes[id] = new_shape

            # Reshape value to correctly calculate output shape
            if values[id] is not None:
                values[id] = np.ma.reshape(values[id], new_shape)

    extended_shapes = [np.ma.concatenate((np.ma.ones(max_dims - len(s), dtype=np.int64), s)) for s in shapes]
    output_shape = extended_shapes[0]
    for si in range(1, len(extended_shapes)):
        for ei in range(max_dims):
            output_shape[ei] = broadcast_dims(output_shape[ei], extended_shapes[si][ei])

    node.out_port(0).data.set_shape(output_shape)

    if node.has_and_set('stop_value_propagation'):
        return

    if op is None or any([v is None for v in values]):
        return

    if len(values) <= 2:
        node.out_port(0).data.set_value(op(*values, **kwargs))
    else:
        node.out_port(0).data.set_value(values[0])
        for i in range(len(values) - 1):
            node.out_port(0).data.set_value(op(node.out_node().value, values[i + 1]))


def bias_add_infer(node, op):
    if node.in_port(0).data.get_value() is not None and node.in_port(1).data.get_value() is not None and op is not None:
        node.out_port(0).data.set_value(op(node.in_port(0).data.get_value(), node.in_port(1).data.get_value()))
    else:
        node.out_port(0).data.set_shape(node.in_port(0).data.get_shape())
