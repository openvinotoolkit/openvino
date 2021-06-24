# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from mo.back.replacement import BackReplacementPattern
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Graph
from mo.ops.const import Const


class GroupedConvWeightsNormalize(BackReplacementPattern):
    """
    This pass is a workaround for nGraph GroupedConvolution operation
    It requires that weights layout will be next: G*O*I,1,H,W
    """
    enabled = True
    force_clean_up = True

    def pattern(self):
        return dict(
            nodes=[
                ('conv', {'type': 'Convolution', 'group': lambda x: x != 1}),
                ('weights', {'type': 'Const', 'kind': 'op'}),
                ('weights_data', {'kind': 'data'}),
            ],
            edges=[('weights', 'weights_data'), ('weights_data', 'conv')]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        conv = match['conv']
        weights = match['weights']
        input_shape = conv.in_port(0).data.get_shape()
        new_weights_shape = int64_array([(weights.value.shape[0] * weights.value.shape[1]) / (input_shape[1] / conv.group), input_shape[1] / conv.group, *weights.value.shape[2:]])
        new_weights = Const(graph, {'value': np.reshape(weights.value, new_weights_shape),
                                    'name':  weights.soft_get('name', weights.id) + '_new'}).create_node()
        weights.out_port(0).get_connection().set_source(new_weights.out_port(0))
        new_weights.infer(new_weights)
