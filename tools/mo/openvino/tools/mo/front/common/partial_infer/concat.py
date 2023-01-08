# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.caffe.extractors.utils import get_canonical_axis_index
from openvino.tools.mo.front.common.partial_infer.utils import shape_array, is_fully_defined, dynamic_dimension
from openvino.tools.mo.ops.op import PermuteAttrs
from openvino.tools.mo.utils.error import Error


def concat_infer(node):
    node_name = node.soft_get('name', node.id)
    if not node.has('axis'):
        N = node.N
        axis_input = node.in_node(N)
        if axis_input.has_valid('value') and axis_input.value.size == 1:
            node['axis'] = axis_input.value.item()
            node.graph.remove_edge(axis_input.node, node.node)  # TODO add skip attribute instead of deleting
        else:
            raise Error('Input with value is not specified for node "{}"'.format(node_name))
    else:
        N = len(node.in_nodes())

    shapes = [node.in_node(i).shape for i in range(N)]
    if any(s is None for s in shapes):
        raise Error('One of the input shapes is not defined for node "{}"'.format(node_name))

    shape = shape_array(shapes[0])

    axis = get_canonical_axis_index(shape, node.axis)
    node.axis = axis

    mask = np.zeros_like(shape, dtype=bool)
    mask[axis] = True  # pylint: disable=unsupported-assignment-operation
    not_mask = np.logical_not(mask)  # pylint: disable=assignment-from-no-return
    for s in shapes[1:]:
        s = shape_array(s)
        if np.ma.allequal(shape[not_mask], s[not_mask]):
            shape[mask] += s[mask]
        else:
            raise Error('Concat input shapes do not match for node "{}" with axis {}'.format(node_name, axis))

    #  dynamic dimensions in the output (except the concat axis) can be deduced from input shape
    for pos in range(len(shape)):
        if shape[pos] is dynamic_dimension and pos != axis:
            for in_shape in shapes:
                if in_shape[pos] is not dynamic_dimension:
                    shape[pos] = in_shape[pos]

    node.out_port(0).data.set_shape(shape)
    PermuteAttrs.create_permute_attrs(node, attrs=[('axis', 'input:0')])

    values = [node.in_node(i).value for i in range(N)]
    if any([v is None for v in values]):
        return

    # if one of the input values are dynamic, the output tensor type is inferred from one of the fully defined inputs
    output_dtype = np.int64
    for input in values:
        if is_fully_defined(input):
            output_dtype = input.dtype

    if any(not is_fully_defined(v) for v in values):
        node.out_port(0).data.set_value(np.ma.concatenate(values, axis=node.axis).astype(output_dtype))
    else:  # there is a serious performance benefit to use concatenation as it is implemented below
        node.out_node(0).value = np.concatenate(values, axis=node.axis).astype(values[0].dtype, copy=False)
        node.out_node(0).shape = shape_array(node.out_node(0).value.shape)
