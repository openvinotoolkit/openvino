# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.elemental import copy_shape_infer
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.op import Op


class PReLU(Op):
    op = 'PReLU'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': self.op,
            'type': self.op,
            'version': 'opset1',

            'infer': self.infer,

            'force_precision_in_ports': {1: 'float'},

            'in_ports_count': 2,
            'out_ports_count': 1,
        }, attrs)

    @staticmethod
    def infer(node):
        if len(node.in_nodes()) == 2:
            gamma_vector = node.in_node(1)
            if np.all(gamma_vector.shape == [1]):
                node['channel_shared'] = 1
            else:
                node['channel_shared'] = 0
            node.in_node(1)['correct_data_type'] = True

        copy_shape_infer(node)
