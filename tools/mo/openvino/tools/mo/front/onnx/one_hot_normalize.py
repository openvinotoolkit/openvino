# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.split_normalizer import SqueezeAxis
from openvino.tools.mo.ops.split import Split
from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Graph


class OneHotNormalize(FrontReplacementSubgraph):
    """
    The ONNX OneHot layer has input with values of shape [2] which contains off and on values. This transformation
    splits this input into two and connects them back to the OneHot layer in reverse order because in OV layer the
    on value goes to port 2 and off values goes to port 3.
    """
    enabled = True

    def run_before(self):
        return [SqueezeAxis]

    def pattern(self):
        return dict(nodes=[('onehot', dict(op='OneHot', split_values=True))],
                    edges=[])

    def replace_sub_graph(self, graph: Graph, match: dict):
        onehot = match['onehot']
        name = onehot.soft_get('name', onehot.id)

        split = create_op_with_const_inputs(graph, Split, {1: np.int64(0)},
                                            {'name': name + '/Split', 'num_splits': 2, 'squeeze_axis': True})

        onehot.in_port(2).get_source().connect(split.in_port(0))
        onehot.in_port(2).disconnect()

        onehot.in_port(3).connect(split.out_port(0))
        onehot.in_port(2).connect(split.out_port(1))
