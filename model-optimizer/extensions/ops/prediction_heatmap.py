# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from mo.graph.graph import Node, Graph
from mo.ops.op import Op


class PredictionHeatmapOp(Op):
    op = 'PredictionHeatmap'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': __class__.op,
            'op': __class__.op,
            'version': 'extension',
            'in_ports_count': 1,
            'out_ports_count': 1,
            'infer': PredictionHeatmapOp.infer
        }
        super().__init__(graph, mandatory_props, attrs)

    @staticmethod
    def infer(node: Node):
        in_node_shape = node.in_nodes()[0].shape.copy()
        top_shape = np.array([1, 1, 1, 1])
        num_person = in_node_shape[0]
        num_joints = in_node_shape[1]
        top_shape[2] = num_person
        top_shape[3] = 3 * num_joints
        node.out_node().shape = top_shape
