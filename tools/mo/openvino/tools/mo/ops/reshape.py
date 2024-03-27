# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import dynamic_dimension, dynamic_dimension_value, is_fully_defined
from openvino.tools.mo.front.extractor import bool_to_str
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.graph.perm_inputs import PermuteInputs
from openvino.tools.mo.ops.op import Op


class Reshape(Op):
    op = 'Reshape'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': self.op,
            'type': self.op,
            'version': 'opset1',

            'infer': self.infer,

            'special_zero': True,
            'reinterp_shape': True,

            'in_ports_count': 2,
            'out_ports_count': 1,
        }, attrs)

    def supported_attrs(self):
        return [('special_zero', lambda node: bool_to_str(node, 'special_zero'))]

    @staticmethod
    def infer(node: Node):
        name = node.soft_get('name', node.id)

        connected_inputs = {idx: port for idx, port in node.in_ports().items() if not port.disconnected()}
        assert len(connected_inputs) == 2 and all([i in connected_inputs for i in range(2)]), \
            "Reshape should have 2 connected input ports, but it doesn't for node: `{}`. Ports: {}" \
            "".format(name, connected_inputs)

        input_shape = node.in_port(0).data.get_shape()
        assert input_shape is not None

        new_shape = node.in_port(1).data.get_value()
        assert new_shape is not None, 'Dynamic Reshape second input is not supported. Node {}'.format(name)

        assert np.argwhere(new_shape == -1).size <= 1, \
            'Reshape second input should not have several `-1` values set. ' \
            'Node: {}, reshape second input value {}'.format(name, new_shape)

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

        # input_shape = [dynamic, 5, 6], new_shape = [0, -1] => output_shape [dynamic, 30]
        # marker that no dynamic input dimensions or all of them are copied with "0" magic value
        all_dynamic_dimension_are_copied = True
        if not is_fully_defined(input_shape):
            for index, x in enumerate(input_shape):
                if x is dynamic_dimension:
                    if index >= len(new_shape) or new_shape[index] != 0:
                        all_dynamic_dimension_are_copied = False

        undefined_dim = dynamic_dimension
        if num_of_output_elements is not dynamic_dimension and all_dynamic_dimension_are_copied and \
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
               "Number of elements in input {} and output {} of reshape node {} mismatch" \
               "".format(input_shape, output_shape, name)

        PermuteInputs().set_input_permutation(node.in_node(1), node, 'output:0', 'shape')

        if node.in_port(0).data.get_value() is not None and is_fully_defined(output_shape):
            node.out_port(0).data.set_value(node.in_port(0).data.get_value().reshape(output_shape))
        else:
            node.out_port(0).data.set_shape(output_shape)
