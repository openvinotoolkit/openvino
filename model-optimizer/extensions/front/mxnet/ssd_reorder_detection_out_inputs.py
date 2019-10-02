"""
 Copyright (c) 2017-2019 Intel Corporation

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

import networkx as nx

from mo.graph.graph import Graph
from mo.front.common.replacement import FrontReplacementPattern
from extensions.front.mxnet.ssd_pattern_remove_transpose import SsdPatternRemoveTranspose
from extensions.front.mxnet.ssd_pattern_flatten_softmax_activation import SsdPatternFlattenSoftmaxActivation


class SsdReorderDetectionOutInputs(FrontReplacementPattern):

    enabled = True

    def run_before(self):
        return [SsdPatternFlattenSoftmaxActivation, SsdPatternRemoveTranspose]

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('multi_box_detection', dict(op='_contrib_MultiBoxDetection'))
            ],
            edges=[])

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        """
        DetectionOutput layer has another order of inputs unlike mxnet.
        Need to reorder _contrib_MultiBoxDetection inputs
        for correct conversion to DetectionOutput layer.

        Parameters
        ----------
        graph : Graph
           Graph with loaded model.
        """
        multi_box_detection_node = match['multi_box_detection']
        conf_node = multi_box_detection_node.in_node(0)
        loc_node = multi_box_detection_node.in_node(1)

        conf_edge_data = graph.get_edge_data(conf_node.id, multi_box_detection_node.id)
        conf_out_port = conf_edge_data[0]['out']
        conf_in_port = conf_edge_data[0]['in']

        loc_edge_data = graph.get_edge_data(loc_node.id, multi_box_detection_node.id)
        loc_out_port = loc_edge_data[0]['out']
        loc_in_port = loc_edge_data[0]['in']

        graph.remove_edge(conf_node.id, multi_box_detection_node.id)
        graph.remove_edge(loc_node.id, multi_box_detection_node.id)

        graph.create_edge(loc_node, multi_box_detection_node, in_port=conf_in_port, out_port=conf_out_port)
        graph.create_edge(conf_node, multi_box_detection_node, in_port=loc_in_port, out_port=loc_out_port)
