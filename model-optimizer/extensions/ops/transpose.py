# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from mo.graph.graph import Graph
from mo.ops.op import Op


class Transpose(Op):
    op = 'Transpose'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': __class__.op,
            'op': __class__.op,
            'version': 'opset1',
            'infer': self.infer,
            'force_precision_in_ports': {1: 'int64'},
            'in_ports_count': 2,
            'out_ports_count': 1,
        }, attrs)

    @staticmethod
    def infer(node):
        # order parameter calculation and checks
        in_ports = node.in_ports()
        connected_ports = [port for port in in_ports.values() if not port.disconnected()]
        input_shape = node.in_port(0).data.get_shape()

        if node.has_and_set('reverse_order'):
            assert len(connected_ports) == 1 and 0 in in_ports, \
                'Cannot infer `{}` due to both order and reverse_order was set'.format(node.soft_get('name'))
            order = np.arange(len(input_shape))[::-1]  # Reverse order
        else:
            # we import PermuteInputs locally because it uses Transpose inside and we have recursive imports
            from mo.graph.perm_inputs import PermuteInputs
            assert len(connected_ports) == 2 and 0 in in_ports and 1 in in_ports, \
                "{} node `{}` should have 2 input ports, where 0-input is a data input and 1-input represents " \
                "Transpose `order`".format(node.op, node.id)
            order = node.in_port(1).data.get_value()
            assert order is not None, 'Cannot infer `{}` because order is None'.format(node.soft_get('name'))
            PermuteInputs().set_input_permutation(node.in_node(1), node, 'input:0', 'order')

        # setting shape and value if applicable
        if node.in_port(0).data.get_value() is not None:
            node.out_port(0).data.set_value(np.transpose(node.in_port(0).data.get_value(), axes=order))
        else:
            node.out_port(0).data.set_shape(input_shape[order])


