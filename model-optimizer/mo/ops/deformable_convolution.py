# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.graph.graph import Graph
from mo.ops.convolution import Convolution
from mo.ops.op import Op


class DeformableConvolution(Op):
    op = 'DeformableConvolution'

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': __class__.op,
            'op': __class__.op,
            'version': 'opset8',
            'infer': Convolution.infer,
            'group': 1,
            'deformable_group': 1,
            'multiplication_transparent': True,
            'multiplication_transparent_ports': [(0, 0), (2, 0)],
            'in_ports_count': 3,
            'out_ports_count': 1,
            'use_bilinear_interpolation_padding': False,
        }, attrs)

    def backend_attrs(self):
        # the same attributes as in a regular convolution and one additional attribute 'deformable_group' and 'group'
        return Convolution(self.graph, {}).backend_attrs() + ['deformable_group', 'group']
