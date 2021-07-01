# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy

from mo.graph.graph import Node, Graph
from mo.ops.op import Op


class SpatialTransformOp(Op):
    op = 'SpatialTransformer'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': __class__.op,
            'op': __class__.op,
            'version': 'experimental',
            'in_ports_count': 1,
            'out_ports_count': 1,
            'infer': SpatialTransformOp.sp_infer
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return [
            'transform_type',
            'sampler_type',
            'output_H',
            'output_W',
            'to_compute_dU',
            'theta_1_1',
            'theta_1_2',
            'theta_1_3',
            'theta_2_1',
            'theta_2_2',
            'theta_2_3'
        ]

    @staticmethod
    def sp_infer(node: Node):
        input_shape = node.in_node(0).shape
        output_shape = copy.copy(input_shape)
        if node.has_valid('output_H'):
            output_shape[2] = node.output_H
        if node.has_valid('output_W'):
            output_shape[3] = node.output_W
        node.out_node().shape = output_shape
