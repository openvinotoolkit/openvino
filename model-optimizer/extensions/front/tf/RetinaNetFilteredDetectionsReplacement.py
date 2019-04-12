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

from extensions.ops.DetectionOutput import DetectionOutput
from extensions.ops.splitv import SplitV
from mo.front.subgraph_matcher import SubgraphMatch
from mo.front.tf.replacement import FrontReplacementFromConfigFileSubGraph
from mo.graph.graph import Node, Graph
from mo.ops.concat import Concat
from mo.ops.const import Const
from mo.ops.eltwise import Eltwise
from mo.ops.power import Power
from mo.ops.reshape import Reshape


class RetinaNetFilteredDetectionsReplacement(FrontReplacementFromConfigFileSubGraph):
    """
    The class replaces the sub-graph that performs boxes post-processing and NMS with the DetectionOutput layer.

    The post-processing in the RetinaNet topology is performed differently from the DetectionOutput layer implementation
    in the Inference Engine. The first one calculates (d_x1, d_y1, d_x2, d_y2) which are a factor of the prior box width
    and height. The DetectionOuput with "code_type" equal to "caffe.PriorBoxParameter.CORNER" just adds predicted deltas
    to the prior box coordinates. This replacer add nodes which calculate prior box widths and heights, apply variances
    to the predicated box coordinates and multiply them. With this approach the DetectionOutput layer with "code_type"
    equal to "caffe.PriorBoxParameter.CORNER" produces the same result as the post-processing in the original topology.
    """
    replacement_id = 'RetinaNetFilteredDetectionsReplacement'

    @staticmethod
    def _create_sub(graph: Graph, input_1: Node, port_1: int, input_2: Node, port_2: int):
        negate = Power(graph, dict(scale=-1, name=input_2.name + '/negate_'))
        add = Eltwise(graph, dict(operation='sum', name=input_1.name + '/add_'))
        out_node = add.create_node([(input_1, port_1), negate.create_node([(input_2, port_2)])])
        return out_node

    def output_edges_match(self, graph: Graph, match: SubgraphMatch, new_sub_graph: dict):
        return {match.output_node(0)[0].id: new_sub_graph['detection_output_node'].id}

    def nodes_to_remove(self, graph: Graph, match: SubgraphMatch):
        new_nodes_to_remove = match.matched_nodes_names()
        new_nodes_to_remove.remove(match.single_input_node(0)[0].id)
        new_nodes_to_remove.remove(match.single_input_node(1)[0].id)
        new_nodes_to_remove.remove(match.single_input_node(2)[0].id)
        return new_nodes_to_remove

    def generate_sub_graph(self, graph: Graph, match: SubgraphMatch):
        reshape_classes_op = Reshape(graph, {'dim': np.array([0, -1])})
        reshape_classes_node = reshape_classes_op.create_node([match.single_input_node(1)[0]],
                                                              dict(name='do_reshape_classes'))

        priors_node = match.single_input_node(2)[0]

        placeholder = [Node(graph, node_id) for node_id in graph.nodes() if Node(graph, node_id).op == 'Placeholder'][0]
        im_height = placeholder.shape[1]
        im_width = placeholder.shape[2]

        # scale prior boxes to the [0, 1] interval
        priors_scale_const_node = Const(graph, {'value': np.array([1 / im_width,
                                                                   1 / im_height,
                                                                   1 / im_width,
                                                                   1 / im_height])}).create_node([])
        priors_scale_node = Eltwise(graph, {'name': 'scale_priors', 'operation': 'mul'}).create_node(
            [priors_node, priors_scale_const_node])

        # calculate prior boxes widths and heights
        split_node = SplitV(graph, {'axis': 2, 'size_splits': [1, 1, 1, 1], 'out_ports_count': 4}).create_node([priors_scale_node])
        priors_width_node = __class__._create_sub(graph, split_node, 2, split_node, 0)
        priors_height_node = __class__._create_sub(graph, split_node, 3, split_node, 1)

        # concat weights and heights into a single tensor and multiple with the box coordinates regression values
        concat_width_height_node = Concat(graph, {'name': 'concat_priors_width_height', 'axis': -1, 'in_ports_count': 4}).create_node(
            [priors_width_node, priors_height_node, priors_width_node, priors_height_node])
        applied_width_height_regressions_node = Eltwise(graph, {'name': 'final_regressions', 'operation': 'mul'}). \
            create_node([concat_width_height_node, match.single_input_node(0)[0]])

        # reshape to 2D tensor as Inference Engine Detection Output layer expects
        reshape_regression_op = Reshape(graph, {'dim': np.array([0, -1])})
        reshape_regression_node = reshape_regression_op.create_node([applied_width_height_regressions_node],
                                                                    {'name': 'reshape_regression'})

        detection_output_op = DetectionOutput(graph, match.custom_replacement_desc.custom_attributes)
        detection_output_op.attrs['old_infer'] = detection_output_op.attrs['infer']
        detection_output_op.attrs['infer'] = __class__.do_infer
        detection_output_node = detection_output_op.create_node(
            [reshape_regression_node, reshape_classes_node, priors_scale_node],
            dict(name=detection_output_op.attrs['type'], clip=1, normalized=1, variance_encoded_in_target=0))

        return {'detection_output_node': detection_output_node}

    @staticmethod
    def do_infer(node):
        # append variances to the tensor with boxes regressions
        prior_boxes = node.in_node(2).value
        assert prior_boxes is not None, "The prior boxes are not constants"
        if prior_boxes is not None:
            variances = np.tile(node.variance, [prior_boxes.shape[-2], 1])
            prior_boxes = prior_boxes.reshape([-1, 4])
            prior_boxes = np.concatenate((prior_boxes, variances), 0)
            #  adding another dimensions, as the prior-boxes are expected as 3d tensor
            prior_boxes = prior_boxes.reshape((1, 2, -1))
            node.in_node(2).shape = np.array(prior_boxes.shape, dtype=np.int64)
            node.in_node(2).value = prior_boxes

        node.old_infer(node)
