# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from openvino.tools.mo.front.common.partial_infer.elemental import copy_shape_infer
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.op import Op


class LayerNorm(Op):
    """
    MXNet operation which normalizes the channels of the input tensor by mean and variance, and applies a scale gamma
    and offset beta. Operation computes output with the same shape as input as following formula:

    out = ((data - mean(data, axis)) / sqrt(var(data, axis) + eps)) * gamma + beta

    inputs:
    data - input data
    gamma - gamma array
    beta - beta array

    attributes:
    axis - axis to perform layer normalization
    eps - epsilon parameter to prevent division by zero
    """

    op = 'LayerNorm'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': self.op,
            'type': None,
            'axis': -1,
            'epsilon': 0.001,
            'infer': copy_shape_infer,
            'in_ports_count': 3,
            'out_ports_count': 1,
        }, attrs)
