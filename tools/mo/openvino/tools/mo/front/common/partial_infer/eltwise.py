# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import dynamic_dimension, dynamic_dimension_value, \
    undefined_shape_of_rank, compatible_shapes, compatible_dims
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.utils.error import Error


def eltwise_infer(node: Node, op=None, **kwargs):
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


def eltwise_reverse_infer(node: Node):
    input_1_shape = node.in_port(0).data.get_shape()
    input_2_shape = node.in_port(1).data.get_shape()
    if input_1_shape is not None and input_2_shape is not None:
        return

    output_shape = node.out_port(0).data.get_shape()
    node_name = node.soft_get('name', node.id)

    if node['auto_broadcast'] is 'none':
        # input_1, input_2 and output shapes must match
        # therefore undefined partial shapes can be exactly defined from output shape
        if output_shape is not None:
            most_defined_shape = output_shape

            # if out_shape = [4, dyn] and input_1_shape = [dyn, 13]
            # then missing shape must be [4, 13]
            if input_1_shape is not None and not compatible_shapes(output_shape, input_1_shape):
                raise Error("shapes are not compatible for node '{}'".format(node_name))
            elif input_1_shape is not None:
                most_defined_shape = find_common_partial_shape(output_shape, input_1_shape)

            if input_2_shape is not None and not compatible_shapes(output_shape, input_2_shape):
                raise Error("shapes are not compatible for node '{}'".format(node_name))
            elif input_2_shape is not None:
                most_defined_shape = find_common_partial_shape(most_defined_shape, input_2_shape)

            if input_1_shape is None:
                node.in_port(0).data.set_shape(most_defined_shape)
            if input_2_shape is None:
                node.in_port(1).data.set_shape(most_defined_shape)
    elif node['auto_broadcast'] == 'numpy':
        if output_shape is not None:
            out_rank = len(output_shape)
            deduced_in_shape = undefined_shape_of_rank(out_rank)

            if input_1_shape is not None and input_2_shape is None and out_rank > len(input_1_shape):
                in_port_to_update = 1
                defined_in_shape = input_1_shape
            elif input_2_shape is not None and input_1_shape is None and out_rank > len(input_2_shape):
                in_port_to_update = 0
                defined_in_shape = input_2_shape
            else:
                return
            defined_in_rank = len(defined_in_shape)

            for i in range(-1, -defined_in_rank - 1, -1):
                assert defined_in_shape[i] == 1 or np.ma.is_masked(defined_in_shape[i]) \
                       or compatible_dims(defined_in_shape[i], output_shape[i]), \
                    "Shapes of Elementwise node '{}' are not compatible for reverse_infer.".format(node_name)

                # if defined_input_shape = [1] and output_shape = [N, 400, 400, 3]
                # partial shape information about sizes should not be lost
                if defined_in_shape[i] == 1 or output_shape[i] == 1:
                    deduced_in_shape[i] = output_shape[i]
            deduced_in_shape[:-defined_in_rank] = output_shape[:-defined_in_rank]

            node.in_port(in_port_to_update).data.set_shape(deduced_in_shape)


def bias_add_infer(node, op):
    if node.in_port(0).data.get_value() is not None and node.in_port(1).data.get_value() is not None and op is not None:
        node.out_port(0).data.set_value(op(node.in_port(0).data.get_value(), node.in_port(1).data.get_value()))
    else:
        node.out_port(0).data.set_shape(node.in_port(0).data.get_shape())


def find_common_partial_shape(shape1, shape2):
    """
    Extracts maximum information from partially defined shapes.

    For example, if shape_1 = [4, dyn] and shape_2 = [dyn, 13]
    then resulting shape will be [4, 13]
    :param shape1: partially defined shape
    :param shape2: partially defined shape
    :return:
    """
    assert compatible_shapes(shape1, shape2), 'shapes must be compatible'
    res = shape1.copy()
    for i, (d1, d2) in enumerate(zip(shape1, shape2)):
        if np.ma.is_masked(res[i]):
            res[i] = d2
    return res
