# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.common.partial_infer.elemental import copy_shape_infer
from mo.graph.graph import Graph
from mo.ops.op import Op


class LayerNorm(Op):
    """
    MXNet operation which normalizes the channels of the input tensor by mean and variance, and applies a scale gamma
    and offset beta
    """

    op = 'LayerNorm'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': __class__.op,
            'type': None,
            'axis': -1,
            'epsilon': 0.001,
            'infer': copy_shape_infer,
            'in_ports_count': 3,
            'out_ports_count': 1,
        }, attrs)
