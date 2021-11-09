# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from mo.front.common.partial_infer.utils import dynamic_dimension, dynamic_dimension_value, is_fully_defined, shape_array
from mo.graph.graph import Node, Graph
from mo.ops.op import Op


class SparseReshape(Op):
    """
    SparseReshape operation reshapes a sparse tensor. It recomputes indices for a new dense shape.
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
        assert np.count_nonzero(new_shape == -1) <= 1, \
            "Value -1 occurs in new shape value more than once"

        num_of_input_elements = np.prod(input_shape)
        num_of_output_elements = 1
        for index, x in enumerate(new_shape):
            if x is dynamic_dimension:
                num_of_output_elements = dynamic_dimension_value
            elif x == 0 and node.has_and_set('special_zero'):
                if input_shape[index] is not dynamic_dimension:
                    num_of_output_elements *= input_shape[index]
            elif x != -1:
                num_of_output_elements *= x

        # input_shape = [dyn, 5, 6], new_shape = [0, -1] => output_shape [dyn, 30]
        dyns_are_copied = True  # true if there are no dyns or all of them are copied with "0" magic value
        if not is_fully_defined(input_shape):
            for index, x in enumerate(input_shape):
                if x is dynamic_dimension:
                    if index >= len(new_shape) or new_shape[index] != 0:
                        dyns_are_copied = False

        undefined_dim = dynamic_dimension
        if num_of_output_elements is not dynamic_dimension and dyns_are_copied and \
                is_fully_defined(new_shape):
            undefined_dim = num_of_input_elements // num_of_output_elements
        output_shape = []
        for index, x in enumerate(new_shape):
            if x == 0 and node.has_and_set('special_zero'):
                output_shape.append(input_shape[index])
            elif x == -1:
                output_shape.append(undefined_dim)
            else:
                output_shape.append(x)

        # even if the new_shape contains some dynamic values we can calculate the actual value by deducing it from the
        # input shape if it is static: input_shape = [5, 3, 8], new_shape = [4, d] => output_shape = [4, 30]
        if is_fully_defined(input_shape) and not is_fully_defined(new_shape):
            dynamic_indices = np.argwhere([item is dynamic_dimension for item in new_shape])
            num_of_output_elements = 1
            if dynamic_indices.size == 1:
                for index, x in enumerate(new_shape):
                    if x == 0 and node.has_and_set('special_zero'):
                        num_of_output_elements *= input_shape[index]
                    elif x is not dynamic_dimension and x != -1:
                        num_of_output_elements *= x
            assert num_of_input_elements % num_of_output_elements == 0, \
                'Incorrect number of output elements deduced for node {}: '.format(name)
            output_shape[dynamic_indices[0][0]] = num_of_input_elements // num_of_output_elements

        assert not is_fully_defined(input_shape) or not is_fully_defined(output_shape) or \
               np.prod(input_shape) == np.prod(output_shape), \
            "Number of elements in input {} and output {} of dynamic reshape node {} mismatch" \
            "".format(input_shape, output_shape, name)

        node.out_port(1).data.set_value(shape_array(output_shape))
        output_indices_shape = np.concatenate((input_indices_shape[0:1], new_shape_shape))
        node.out_port(0).data.set_shape(output_indices_shape)

        # TODO: implement constant value propagation for common case
        if np.array_equal(input_shape, output_shape) and input_indices_value is not None:
            node.out_port(0).data.set_value(input_indices_value)
