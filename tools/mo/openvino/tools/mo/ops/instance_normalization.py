# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.op import Op


class InstanceNormalization(Op):
    ''' InstanceNormalization like it is defined in ONNX

        y = scale * (x - mean) / sqrt(variance + epsilon) + B

        where x is input(0), scale is input(1) and B is input(2)
    '''
    op = 'InstanceNormalization'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': __class__.op,
            'epsilon': None,
            #'infer' - is not needed, this op should be replaced by a front replacer
        }, attrs)

    def supported_attrs(self):
        return ['epsilon']
