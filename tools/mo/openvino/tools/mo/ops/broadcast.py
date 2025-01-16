# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.utils import is_fully_defined, shape_array, undefined_shape_of_rank
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.graph.perm_inputs import PermuteInputs
from openvino.tools.mo.ops.op import Op
from openvino.tools.mo.utils.broadcasting import bi_directional_shape_broadcasting, uni_directional_shape_broadcasting, \
    uni_directional_broadcasting, bi_directional_broadcasting, explicit_broadcasting, explicit_shape_broadcasting
from openvino.tools.mo.utils.error import Error


class Broadcast(Op):
    """ Broadcast tensor to a given shape with optional axis parameter

        Inputs:
            [0] - tensor to be broadcasted
            [1] - shape to be broadcast to
            [2] - optional axes_mapping tensor
    """

    op = 'Broadcast'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': self.op,
            'op': self.op,
            'version': 'opset3',
            'mode': 'numpy',
            'in_ports_count': 3,
            'out_ports_count': 1,
            'force_precision_in_ports':
                {1: 'int64',
                 2: 'int64',
                 },
            'infer': self.infer,
        }, attrs)

    def supported_attrs(self):
        return ['mode']

    @staticmethod
    def infer(node: Node):
        node_name = node.soft_get('name', node.id)

        input_shape = node.in_port(0).data.get_shape()
        input_value = node.in_port(0).data.get_value()
        target_shape_shape = node.in_port(1).data.get_shape()
        target_shape = node.in_port(1).data.get_value()
        assert node.has_and_set('mode'), 'Broadcasting mode is not defined for node "{}"'.format(node_name)

        PermuteInputs().set_input_permutation(node.in_node(1), node, 'output:0', 'shape')

        # Dynamic target shape is possible to infer only if shape of target shape is static
        if target_shape is None:
            assert len(target_shape_shape) == 1, 'Shape of target_shape must be [1] for node "{}"'.format(node_name)
            assert is_fully_defined(target_shape_shape), 'Output shape is not defined for node "{}"'.format(node_name)
            new_shape = undefined_shape_of_rank(target_shape_shape.item(0))
            node.out_port(0).data.set_shape(new_shape)            
            if node.mode == 'explicit':
                assert node.is_in_port_connected(
                    2), 'Axes mapping must be specified for Broadcast(mode="explicit"). Node: `{}`'.format(node_name)
                PermuteInputs().set_input_permutation(node.in_node(2), node, 'output:0', 'axis')
            return

        if input_value is not None and not node.has_and_set('stop_value_propagation') and \
                is_fully_defined(target_shape):
            if node.mode == 'numpy':
                node.out_port(0).data.set_value(uni_directional_broadcasting(input_value, target_shape))
            elif node.mode == 'bidirectional':
                node.out_port(0).data.set_value(bi_directional_broadcasting(input_value, target_shape))
            elif node.mode == 'explicit':
                axes_mapping = node.in_port(2).data.get_value()
                assert axes_mapping is not None, 'Broadcast(mode="explicit") with dynamic axes_mapping input ' \
                                                 'is not supported. Node: `{}`'.format(node_name)
                PermuteInputs().set_input_permutation(node.in_node(2), node, 'output:0', 'axis')
                axes_mapping = node.in_port(2).data.get_value()
                node.out_port(0).data.set_value(explicit_broadcasting(input_value, target_shape, axes_mapping))
            else:
                raise Error('The node "{}" has unsupported mode "{}"'.format(node_name, node.mode))
        else:
            if node.mode == 'numpy':
                node.out_port(0).data.set_shape(uni_directional_shape_broadcasting(input_shape, target_shape))
            elif node.mode == 'bidirectional':
                node.out_port(0).data.set_shape(bi_directional_shape_broadcasting(input_shape, target_shape))
            elif node.mode == 'explicit':
                axes_mapping = node.in_port(2).data.get_value()
                assert axes_mapping is not None, 'Broadcast(mode="explicit") with dynamic axes_mapping input ' \
                                                 'is not supported. Node: `{}`'.format(node_name)
                PermuteInputs().set_input_permutation(node.in_node(2), node, 'output:0', 'axis')
                axes_mapping = node.in_port(2).data.get_value()
                new_shape, _ = explicit_shape_broadcasting(input_shape, target_shape, axes_mapping)
                node.out_port(0).data.set_shape(new_shape)
            else:
                raise Error('The node "{}" has unsupported mode "{}"'.format(node_name, node.mode))
