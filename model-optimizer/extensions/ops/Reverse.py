# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Graph
from mo.ops.op import Op


class Reverse(Op):
    op = 'Reverse'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            # 'type': __class__.op, # Internal MO primitive
            'axis': None,
            'op': __class__.op,
            'in_ports_count': 2,
            'out_ports_count': 1,
            'infer': __class__.infer,
        }
        super().__init__(graph, mandatory_props, attrs)

    @staticmethod
    def infer(node):
        input_data_shape = node.in_node(0).shape
        assert input_data_shape is not None
        if not node.has_valid('axis'):
            assert 1 in node.in_nodes()
            assert node.in_node(1).has_valid('value')
            assert node.in_node(1).value.size == 1

            node['axis'] = node.in_node(1).value.item()
            node.in_port(1).disconnect()

        assert node.has_valid('axis')

        assert len(node.out_nodes()) == 1
        node.out_node().shape = input_data_shape.copy()
        if node.in_node().value is not None:
            node.out_node().value = np.flip(node.in_node().value, node['axis'])
            assert np.array_equal(int64_array(node.out_node().value.shape), input_data_shape)
