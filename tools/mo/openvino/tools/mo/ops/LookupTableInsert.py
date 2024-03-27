# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.op import Op


class LookupTableInsert(Op):
    """
    This operation has only output control flow edges and no output data edges in some models.
    And for these cases implementation of the shape inference is needed since the shape inference is executed
    before control flow edges resolving. This operation has non-tensor output so the output shape is empty.
    """
    enabled = False
    op = 'LookupTableInsert'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': None,
            'op': self.op,
            'infer': self.infer,
            'in_ports_count': 3,
            'out_ports_count': 1,
        }
        super().__init__(graph, mandatory_props, attrs)

    @staticmethod
    def infer(node: Node):
        node_name = node.soft_get('name', node.id)
        connected_in_ports = [port for port in node.in_ports().values() if not port.disconnected()]
        assert len(connected_in_ports) == 3, \
            "Incorrect number of inputs for {} node".format(node_name)

        # check shapes of input tensors
        keys_shape = node.in_port(1).data.get_shape()
        values_shape = node.in_port(2).data.get_shape()
        assert np.array_equal(keys_shape, values_shape), \
            'Shapes of tensors with keys and values must be equal for {} node'.format(node_name)

        # set output shape that must be empty
        # since output is not a tensor
        node.out_port(0).data.set_shape([])
