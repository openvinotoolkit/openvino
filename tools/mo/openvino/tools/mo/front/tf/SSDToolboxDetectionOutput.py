# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.DetectionOutput import DetectionOutput
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.subgraph_matcher import SubgraphMatch
from openvino.tools.mo.front.tf.replacement import FrontReplacementFromConfigFileSubGraph
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.const import Const
from openvino.tools.mo.ops.op import PermuteAttrs
from openvino.tools.mo.ops.reshape import Reshape
from openvino.tools.mo.ops.result import Result


class SSDToolboxDetectionOutputReplacement(FrontReplacementFromConfigFileSubGraph):
    replacement_id = 'SSDToolboxDetectionOutput'

    def nodes_to_remove(self, graph: Graph, match: SubgraphMatch):
        return []

    def generate_sub_graph(self, graph: Graph, match: SubgraphMatch):
        # OV DetectionOutput layer consumes flattened confidences and locations tensors.
        # That is why we add reshapes before them.
        locs_node = match.single_input_node(0)
        conf_node = match.single_input_node(1)
        prior_boxes_node = match.single_input_node(2)

        locs_out_nodes = locs_node[0].out_nodes()
        assert len(locs_out_nodes) == 1
        locs_out_node = locs_out_nodes[list(locs_out_nodes.keys())[0]]
        assert locs_out_node.op == "Result", locs_out_node.op
        graph.remove_node(locs_out_node.id)

        conf_out_nodes = conf_node[0].out_nodes()
        assert len(conf_out_nodes) == 1
        conf_out_node = conf_out_nodes[list(conf_out_nodes.keys())[0]]
        assert conf_out_node.op == "Result", conf_out_node.op
        graph.remove_node(conf_out_node.id)

        # reshape operation to flatten confidence tensor
        const = Const(graph, {'value': int64_array([0, -1])}).create_node()
        reshape_loc_node = Reshape(graph, {}).create_node([locs_node, const], dict(name='DetectionOutput_Reshape_loc_'))

        # reshape operation to flatten confidence tensor
        reshape_conf_node = Reshape(graph, {}).create_node([conf_node, const], dict(name='DetectionOutput_Reshape_conf_'))

        # remove the Result node after the priors node
        assert prior_boxes_node[0].out_node().op == "Result"
        graph.remove_node(prior_boxes_node[0].out_node().id)

        # reshape operation for prior boxes tensor
        const = Const(graph, {'value': int64_array([1, 2, -1])}).create_node()
        reshape_priors_node = Reshape(graph, {}).create_node([prior_boxes_node, const],
                                                             dict(name='DetectionOutput_Reshape_priors_'))
        # create Detection Output node with three inputs: locations, confidences and prior boxes
        detection_output_op = DetectionOutput(graph, match.custom_replacement_desc.custom_attributes)
        detection_output_node = detection_output_op.create_node(
            [reshape_loc_node, reshape_conf_node, reshape_priors_node],
            dict(name=detection_output_op.attrs['type'] + '_'))
        PermuteAttrs.set_permutation(reshape_priors_node, detection_output_node, None)

        # create Output node to mark DetectionOutput as a graph output operation
        output_op = Result(graph)
        output_op.create_node([detection_output_node], dict(name='sink_'))
        return {}
