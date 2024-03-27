# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import dynamic_dimension, is_fully_defined, shape_array, strict_compare_tensors, dynamic_dimension_value
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.op import Op


class SparseReshape(Op):
    """
    SparseReshape operation reshapes a sparse tensor in Coordinate list (COO) format
    It recomputes indices for a new dense shape.
    """
    op = 'SparseReshape'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': None,
            'op': __class__.op,
            'infer': self.infer,
            'in_ports_count': 3,
            'out_ports_count': 2,
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return []

    @staticmethod
    def infer(node: Node):
        name = node.soft_get('name', node.id)

        input_indices_shape = node.in_port(0).data.get_shape()
        input_indices_value = node.in_port(0).data.get_value()
        input_shape = node.in_port(1).data.get_value()
        new_shape = node.in_port(2).data.get_value()
        new_shape_shape = node.in_port(2).data.get_shape()

        assert input_shape is not None and new_shape is not None, \
            "Values for input shape and new shape must be defined"
        assert len(np.argwhere(new_shape == -1)) <= 1, \
            "Value -1 occurs in new shape value more than once"
        assert len(np.argwhere(new_shape < -1)) == 0, \
            "Only non-negative or -1 values are allowed"

        output_shape = np.ma.masked_array(new_shape, mask=new_shape == -1, fill_value=dynamic_dimension_value)
        assert not is_fully_defined(input_shape) or not is_fully_defined(output_shape) or \
               np.prod(input_shape) == np.prod(output_shape), \
            "Number of elements in input {} and output {} of dynamic reshape node {} mismatch" \
            "".format(input_shape, output_shape, name)

        # we can deduce -1 only if input_shape is fully defined and
        # there is one dynamic dimension in output_shape
        if is_fully_defined(input_shape) and np.ma.count_masked(output_shape) == 1:
            undefined_dim_size = np.prod(input_shape) // np.prod(output_shape)

            undefined_idx = np.where(output_shape == dynamic_dimension)[0][0]
            output_shape[undefined_idx] = undefined_dim_size
            output_shape.mask[undefined_idx] = False

        node.out_port(1).data.set_value(shape_array(output_shape))
        output_indices_shape = np.concatenate((input_indices_shape[0:1], new_shape_shape))
        node.out_port(0).data.set_shape(output_indices_shape)

        # TODO: implement constant value propagation for common case with scipy.sparse.coo_matrix.reshape
        # instead of compatible_shapes we intentionally use np.array_equal
        if strict_compare_tensors(input_shape, output_shape) and input_indices_value is not None:
            node.out_port(0).data.set_value(input_indices_value)
