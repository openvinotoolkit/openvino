"""
 Copyright (c) 2018-2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import numpy as np

from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern


class TensorFlowLSTMtoGeneric(MiddleReplacementPattern):
    """
    Resolves all differences in TensorFlow LSTMCell and Inference Engine LSTMCell:
    - weights transposing
    - shift_const value addition to biases
    - extra inputs deletion
    """
    enabled = True

    def run_after(self):
        from extensions.middle.pass_separator import MiddleStart
        return [MiddleStart]

    def run_before(self):
        from extensions.middle.permute_tensor_iterator import TransposeTensorIteratorLSTM
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

        # TF stores weights in IO, but IE requires it in OI: transpose
        weights = weights.transpose()

        weights_node.value = weights
        weights_node.shape = np.array(weights.shape, dtype=np.int64)
        biases_node.value = biases
        biases_node.shape = np.array(biases.shape, dtype=np.int64)

        # Cut all extra inputs off
        for i in range(len(node.inputs), len(node.inputs) + len(node.extra_inputs)):
            node.graph.remove_edge(node.in_node(i).id, node.id)
