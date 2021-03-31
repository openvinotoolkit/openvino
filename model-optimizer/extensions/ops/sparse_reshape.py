# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

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
        input_indices_shape = node.in_port(0).data.get_shape()
        input_indices_value = node.in_port(0).data.get_value()
        input_shape_value = node.in_port(1).data.get_value()
        new_shape_value = node.in_port(2).data.get_value()
        new_shape_shape = node.in_port(2).data.get_shape()

        assert input_shape_value is not None and new_shape_value is not None, \
            "Values for input shape and new shape must be defined"
        assert np.count_nonzero(new_shape_value == -1) <= 1, \
            "Value -1 occurs in new shape value more than once"

        node.out_port(1).data.set_shape(new_shape_shape)
        output_shape_value = new_shape_value
        if np.count_nonzero(output_shape_value == -1) == 1:
            elem = np.prod(input_shape_value) // np.prod(new_shape_value[new_shape_value != -1])
            output_shape_value[output_shape_value == -1] = elem
        node.out_port(1).data.set_value(output_shape_value)
        output_indices_shape = np.concatenate((input_indices_shape[0:1], new_shape_shape))
        node.out_port(0).data.set_shape(output_indices_shape)

        # TODO: implement constant value propagation for common case
        if np.array_equal(input_shape_value, output_shape_value) and input_indices_value is not None:
            node.out_port(0).data.set_value(input_indices_value)
