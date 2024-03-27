# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

import numpy as np

from openvino.tools.mo.front.Pack import Pack
from openvino.tools.mo.front.tf.nearest_neighbor_upsampling import NearestNeighborUpsampling
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.const import Const


def is_value_is_constant(val: np.ndarray, const: [int, float]):
    if val.ndim > 1:
        return False
    if val.ndim == 1 and len(val) > 1:
        return False
    return val.item() == const


class FlattenToReshapeableReshape(FrontReplacementSubgraph):
    """
    The TensorFlow implementation of the Flatten operation is not reshape-able because the batch size is hardcoded
    during the constant propagation. This transform sets the 'dim' attribute for the Reshape to [0, -1].
    """
    enabled = True

    def run_after(self):
        return [NearestNeighborUpsampling]

    def run_before(self):
        return [Pack]

    def pattern(self):
        return dict(
            nodes=[
                ('shape', dict(op='ShapeOf')),
                ('strided_slice', dict(op='StridedSlice')),
                ('pack', dict(op='Pack')),
                ('const', dict(op='Const')),
                ('reshape', dict(op='Reshape')),
            ],
            edges=[
                ('shape', 'strided_slice', {'in': 0}),
                ('strided_slice', 'pack', {'in': 0}),
                ('const', 'pack', {'in': 1}),
                ('pack', 'reshape', {'in': 1}),
            ])

    @staticmethod
    def replace_sub_graph(graph: Graph, match: dict):
        strided_slice_node = match['strided_slice']
        const_node = match['const']
        reshape_node = match['reshape']
        pack_node = match['pack']

        if not const_node.has_valid('value') or not is_value_is_constant(const_node.value, -1):
            log.debug('The pattern does not correspond to flatten. The second reshape dimension is not -1. It is {}'.
                      format(const_node.soft_get('value')))
            return
        if len(pack_node.in_nodes()) != 2:
            log.debug('The pattern does not correspond to flatten. The "Pack" operation produces tensor with 3 items '
                      'but should produce just 2.')
            return

        expected_values = [0, 1, 1]  # expected values to a StridedSlice to get the batch size
        for ind in range(3):
            if not strided_slice_node.in_node(ind + 1).has_valid('value') or \
                    not is_value_is_constant(strided_slice_node.in_node(ind + 1).value, expected_values[ind]):
                log.debug('The pattern does not correspond to flatten because of the input with index {}. The value is '
                          '"{}".'.format(ind, strided_slice_node.soft_get('value')))
                return

        reshape_node.in_port(1).disconnect()
        reshape_const_node = Const(graph, {'value': int64_array([0, -1]),
                                           'name': reshape_node.soft_get('name', reshape_node.id) + '/shape'}).create_node()
        reshape_node.in_port(1).connect(reshape_const_node.out_port(0))
        reshape_node['special_zero'] = True
        log.debug('The node "{}" is actually a Flatten node'.format(reshape_node.soft_get('name')))
