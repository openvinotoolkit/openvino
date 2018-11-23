"""
 Copyright (c) 2018 Intel Corporation

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
import numpy as np

from extensions.front.standalone_const_eraser import StandaloneConstEraser
from extensions.ops.DetectionOutput import DetectionOutput
from mo.front.subgraph_matcher import SubgraphMatch
from mo.front.tf.replacement import FrontReplacementFromConfigFileSubGraph
from mo.graph.graph import Node
from mo.ops.op import PermuteAttrs
from mo.ops.output import Output
from mo.ops.reshape import Reshape


class SSDToolboxDetectionOutputReplacement(FrontReplacementFromConfigFileSubGraph):
    replacement_id = 'SSDToolboxDetectionOutput'

    def run_before(self):
        return [StandaloneConstEraser]

    def nodes_to_remove(self, graph: nx.MultiDiGraph, match: SubgraphMatch):
        return []

    def generate_sub_graph(self, graph: nx.MultiDiGraph, match: SubgraphMatch):
        # IE DetectionOutput layer consumes flattened confidences and locations tensors.
        # That is why we add reshapes before them.
        locs_node = match.single_input_node(0)
        conf_node = match.single_input_node(1)
        prior_boxes_node = match.single_input_node(2)

        # reshape operation to flatten confidence tensor
        reshape_loc_op = Reshape(graph, {'dim': np.array([0, -1])})
        reshape_loc_node = reshape_loc_op.create_node([locs_node], dict(name='DetectionOutput_Reshape_loc_'))

        # reshape operation to flatten confidence tensor
        reshape_conf_op = Reshape(graph, {'dim': np.array([0, -1])})
        reshape_conf_node = reshape_conf_op.create_node([conf_node], dict(name='DetectionOutput_Reshape_conf_'))

        # remove the OpOutput node after the priors node
        assert prior_boxes_node[0].out_node().op == "OpOutput"
        graph.remove_node(prior_boxes_node[0].out_node().id)

        # reshape operation for prior boxes tensor
        reshape_priors_op = Reshape(graph, {'dim': np.array([1, 2, -1])})
        reshape_priors_node = reshape_priors_op.create_node([prior_boxes_node],
                                                            dict(name='DetectionOutput_Reshape_priors_'))
        # create Detection Output node with three inputs: locations, confidences and prior boxes
        detection_output_op = DetectionOutput(graph, match.custom_replacement_desc.custom_attributes)
        detection_output_op.attrs['old_infer'] = detection_output_op.attrs['infer']
        detection_output_op.attrs['infer'] = __class__.do_infer
        detection_output_node = detection_output_op.create_node(
            [reshape_loc_node, reshape_conf_node, reshape_priors_node],
            dict(name=detection_output_op.attrs['type'] + '_'))
        PermuteAttrs.set_permutation(reshape_priors_node, detection_output_node, None)

        # create Output node to mark DetectionOutput as a graph output operation
        output_op = Output(graph)
        output_op.create_node([detection_output_node], dict(name='sink_'))
        return {}

    @staticmethod
    def do_infer(node: Node):
        """
        This infer function is used to set attribute 'force_precision' in the data node of the prior boxes because
        it should be in FP32 even if the model has been created in the FP16 or another format.
        """
        node.in_node(2)['force_precision'] = 'FP32'
        node.old_infer(node)
