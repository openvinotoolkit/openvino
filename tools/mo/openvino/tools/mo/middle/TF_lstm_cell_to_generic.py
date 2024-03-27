# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern


class TensorFlowLSTMtoGeneric(MiddleReplacementPattern):
    """
    Resolves all differences in TensorFlow LSTMCell and OpenVINO LSTMCell:
    - weights transposing
    - shift_const value addition to biases
    - extra inputs deletion
    """
    enabled = True

    def run_after(self):
        from openvino.tools.mo.middle.pass_separator import MiddleStart
        return [MiddleStart]

    def run_before(self):
        from openvino.tools.mo.middle.permute_tensor_iterator import TransposeTensorIteratorLSTM
        return [TransposeTensorIteratorLSTM]


    def pattern(self):
        return dict(
            nodes=[('lstm', dict(op='LSTMCell', tf=True))],
            edges=[]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        weights_node = match['lstm'].in_node(3)
        biases_node = match['lstm'].in_node(4)
        node = match['lstm']
        shift_const = node.shift_const

        # make sure that the node is the only consumer or weights and biases
        # to let us modify them without hassle
        assert len(weights_node.out_nodes()) == 1
        assert len(biases_node.out_nodes()) == 1

        # Assign temporary shape for them for easier manipulation
        # TF stores weights in IO order
        input_size = node.in_node(0).shape[1]
        hidden_size = node.in_node(1).shape[1]
        weights = weights_node.value
        biases = biases_node.value
        assert weights.shape[0] == input_size + hidden_size, \
            "weights.shape={} input_size={} hidden_size={}".format(weights.shape, input_size, hidden_size)
        assert weights.shape[1] == biases.shape[0] == 4 * hidden_size, \
            "weights.shape={} biases.shape={} hidden_size={}".format(weights.shape, biases.shape, hidden_size)

        weights = weights.reshape([
            weights.shape[0],
            4,  # gates
            hidden_size
        ])

        biases = biases.reshape([
            4,  # gates
            hidden_size
        ])

        # Reorder gates icfo --> fico for both weights and biases
        gate_reorder = [2, 0, 1, 3]
        weights = np.take(weights, gate_reorder, axis=1)
        biases = np.take(biases, gate_reorder, axis=0)

        # shift_const.value should be added to the first 1/4th part of the biases (f-gate: 0)
        # Note: in case of moving this code up before gate reordering, the addition
        # should be applied at different place
        biases[0] += shift_const

        # Return to the original shapes
        weights = weights.reshape([weights.shape[0], -1])
        biases = biases.flatten()

        # TF stores weights in IO, but OV requires it in OI: transpose
        weights = weights.transpose()

        weights_node.value = weights
        weights_node.shape = int64_array(weights.shape)
        biases_node.value = biases
        biases_node.shape = int64_array(biases.shape)

        # Cut all extra inputs off
        for i in range(len(node.inputs), len(node.inputs) + len(node.extra_inputs)):
            node.graph.remove_edge(node.in_node(i).id, node.id)
