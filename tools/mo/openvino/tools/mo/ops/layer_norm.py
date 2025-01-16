# Copyright (C) 2018-2024 Intel Corporation
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
    output_mean_var - output the mean and std calculated along the given axis. Default value is False. Non default value
                      is not supported
    """

    op = 'LayerNorm'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': self.op,
            'type': None,
            'axis': -1,
            'epsilon': 0.001,
            'output_mean_var': False,
            'infer': copy_shape_infer,
            'in_ports_count': 3 if attrs.get('output_mean_var') is True else 1,
            'out_ports_count': 3,
        }, attrs)
