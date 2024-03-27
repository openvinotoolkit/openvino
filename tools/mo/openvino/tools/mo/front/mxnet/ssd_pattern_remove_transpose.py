# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.mxnet.ssd_pattern_flatten_softmax_activation import SsdPatternFlattenSoftmaxActivation
from openvino.tools.mo.front.mxnet.ssd_pattern_remove_flatten import SsdPatternRemoveFlatten
from openvino.tools.mo.front.mxnet.ssd_pattern_remove_reshape import SsdPatternRemoveReshape
from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph
from openvino.tools.mo.graph.graph import Graph


class SsdPatternRemoveTranspose(FrontReplacementSubgraph):
    enabled = True

    def run_before(self):
        return [SsdPatternFlattenSoftmaxActivation, SsdPatternRemoveFlatten, SsdPatternRemoveReshape]

    def pattern(self):
        return dict(
            nodes=[
                ('transpose', dict(op='Transpose')),
                ('softmax_activation', dict(op='SoftMax')),
                ('multi_box_detection', dict(op='_contrib_MultiBoxDetection'))
            ],
            edges=[
                ('transpose', 'softmax_activation', {'in': 0}),
                ('softmax_activation', 'multi_box_detection', {'in': 1}),
            ]
        )

    def replace_sub_graph(self, graph: Graph, match: dict):
        """
        Need to find each occurrence of pattern:
        transpose -> SoftmaxActivation -> _contrib_MultiBoxDetection
        remove transpose layer to secure the order of weights in SoftMax to be the same as
        OV expects weights to be in following order: class-wise values for each priorbox.
        priorboxes change the quickest

        Parameters
        ----------
        graph : Graph
           Graph with loaded model.
         match : dict
           Patterns which were found in graph structure.
        """
        transpose_node = match['transpose']
        softmax_activation = match['softmax_activation']
        transpose_in_node = transpose_node.in_node(0)

        graph.remove_edge(transpose_in_node.id, transpose_node.id)
        graph.remove_edge(transpose_node.id, softmax_activation.id)
        graph.remove_node(transpose_node.id)
        graph.create_edge(transpose_in_node, softmax_activation)
