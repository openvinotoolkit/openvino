# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.mxnet.ssd_pattern_remove_flatten import SsdPatternRemoveFlatten
from openvino.tools.mo.front.mxnet.ssd_pattern_remove_reshape import SsdPatternRemoveReshape
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph
from openvino.tools.mo.front.tf.graph_utils import create_op_node_with_second_input
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.reshape import Reshape


class SsdPatternFlattenSoftmaxActivation(FrontReplacementSubgraph):
    enabled = True

    def run_before(self):
        return [SsdPatternRemoveFlatten, SsdPatternRemoveReshape]

    def pattern(self):
        return dict(
            nodes=[
                ('softmax_activation', dict(op='SoftMax')),
                ('multi_box_detection', dict(op='_contrib_MultiBoxDetection'))
            ],
            edges=[
                ('softmax_activation', 'multi_box_detection', {'in': 1})
            ]
        )

    def replace_sub_graph(self, graph: Graph, match: dict):
        """
        Need to find the pattern: SoftmaxActivation -> DetectionOutput
        DetectionOutput in OV expects flattened input from SoftMax, that is why there is the need to add
        Flatten layer

        Parameters
        ----------
        graph : Graph
           Graph with loaded model.
         match : dict
           Patterns which were found in graph structure.
        """
        softmax_activation = match['softmax_activation']
        multi_box_detection = match['multi_box_detection']
        softmax_activation['axis'] = -1
        edge_data = graph.get_edge_data(softmax_activation.id, multi_box_detection.id)
        out_port = edge_data[0]['out']
        in_port = edge_data[0]['in']
        graph.remove_edge(softmax_activation.id, multi_box_detection.id)
        new_reshape_node = create_op_node_with_second_input(graph, Reshape, int64_array([0, -1]),
                                                            dict(op='Reshape',
                                                                 name=multi_box_detection.name + '/Reshape_'),
                                                            softmax_activation)
        graph.create_edge(new_reshape_node, multi_box_detection, in_port=in_port, out_port=out_port)
