# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.graph.graph import Node, Graph
from mo.graph.perm_inputs import PermuteInputs
from mo.ops.op import Op
from mo.utils.broadcasting import bi_directional_shape_broadcasting, uni_directional_shape_broadcasting, \
    uni_directional_broadcasting, bi_directional_broadcasting, explicit_broadcasting, explicit_shape_broadcasting
from mo.utils.error import Error


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
            'type': __class__.op,
            'op': __class__.op,
            'version': 'opset3',
            'mode': 'numpy',
            'in_ports_count': 3,
            'out_ports_count': 1,
            'force_precision_in_ports':
                {1: 'int64',
                 2: 'int64',
                 },
            'infer': __class__.infer,
        }, attrs)

    def supported_attrs(self):
        return ['mode']

    @staticmethod
    def infer(node: Node):
        node_name = node.soft_get('name', node.id)

        input_shape = node.in_port(0).data.get_shape()
        input_value = node.in_port(0).data.get_value()
        target_shape = node.in_port(1).data.get_value()
        assert target_shape is not None, 'Output shape is not defined for node "{}"'.format(node_name)
        assert node.has_and_set('mode'), 'Broadcasting mode is not defined for node "{}"'.format(node_name)

        PermuteInputs().set_input_permutation(node.in_node(1), node, 'output:0', 'shape')

        if input_value is not None and not node.has_and_set('stop_value_propagation'):
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
                new_shape,_ = explicit_shape_broadcasting(input_shape, target_shape, axes_mapping)
                node.out_port(0).data.set_shape(new_shape)
            else:
                raise Error('The node "{}" has unsupported mode "{}"'.format(node_name, node.mode))
