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

import logging as log
from collections import deque

import networkx as nx
import numpy as np

from extensions.front.standalone_const_eraser import StandaloneConstEraser
from extensions.front.tf.Preprocessor import PreprocessorReplacement
from extensions.ops.DetectionOutput import DetectionOutput
from mo.front.common.weights import swap_weights_xy
from mo.front.subgraph_matcher import SubgraphMatch
from mo.front.tf.replacement import FrontReplacementFromConfigFileSubGraph
from mo.graph.graph import create_edge, Node
from mo.ops.reshape import Reshape
from mo.ops.softmax import Softmax
from mo.utils.graph import backward_bfs_for_operation


def squeeze_reshape_and_concat(start_nodes: list):
    """
    The function looks for Reshape ops after the 'start_nodes' with 4D output and remove the dimension with index 2
    which should be equal to 1. This is a workaround to make tensor 3D so it's shape will not be transposed during the
    IR generation. The problem arises when bounding boxes predictions are reshaped from [1, 1, 1, X] to
    [1, X / 4, 1, 4]. The result tensor should not be transposed because after transpose it will have shape
    [1, 4, X / 4, 1] and the concatenation over dimension with index 2 will produce incorrect tensor.
    Also the function looks for Concat ops and change the concat dimension from 2 to 1.
    :param start_nodes: list of nodes to start search from.
    :return: None
    """
    q = deque()
    q.extend(start_nodes)
    while len(q) != 0:
        cur_node = q.popleft()
        if cur_node.has_valid('type'):
            if cur_node.type == 'DetectionOutput':  # do not go beyond the DetectionOutput node
                continue
            if cur_node.type == 'Reshape' and len(cur_node.out_node().shape) == 4:
                log.debug("Found Reshape op with 4D output {}".format(cur_node.id))
                if cur_node.in_node(1).has_valid('value') and cur_node.in_node(1).value is not None:
                    new_shape = cur_node.in_node(1).value
                    assert new_shape[2] == 1
                    new_shape = np.delete(new_shape, 2)
                    cur_node.in_node(1).value = new_shape
                    # run infer function once again
                    cur_node.infer(cur_node)
                else:
                    log.warning("The reshape size is not defined!")
            if cur_node.type == 'Concat' and len(cur_node.out_node().shape) == 4:
                log.debug("Found Concat op with 4D output {}".format(cur_node.id))
                cur_node.axis = 1
                # run infer function once again
                cur_node.infer(cur_node)

        out_node_size = len(cur_node.out_nodes())
        for ind in range(out_node_size):
            node = cur_node.out_node(ind)
            q.append(node)


class PostprocessorReplacement(FrontReplacementFromConfigFileSubGraph):
    replacement_id = 'TFObjectDetectionAPIDetectionOutput'

    def run_after(self):
        return [PreprocessorReplacement]

    def run_before(self):
        return [StandaloneConstEraser]

    def output_edges_match(self, graph: nx.DiGraph, match: SubgraphMatch, new_sub_graph: dict):
        # the DetectionOutput in IE produces single tensor, but in TF it produces two tensors, so we need to create only
        # one output edge match
        return {match.output_node(0)[0].id: new_sub_graph['detection_output_node'].id}

    def input_edges_match(self, graph: nx.DiGraph, match: SubgraphMatch, new_sub_graph: dict):
        locs_consumer_node, locs_consumer_node_port = match.input_nodes(0)[0]
        conf_consumer_node, conf_consumer_node_port = match.input_nodes(1)[0]
        priors_consumer_node, priors_consumer_node_port = match.input_nodes(2)[0]
        # create matching nodes for locations and confidence tensors using simple scheme "old_node_name: new_node_name"
        # which in fact means "(old_node_name, 0): (new_node_name, 0)", while first '0' means old_port and the second
        # zero defines 'new_port'.
        return {locs_consumer_node.id: new_sub_graph['reshape_loc_node'].id,
                conf_consumer_node.id: new_sub_graph['reshape_conf_node'].id,
                priors_consumer_node.id: (new_sub_graph['detection_output_node'].id, 2),
                }

    def generate_sub_graph(self, graph: nx.MultiDiGraph, match: SubgraphMatch):
        print('WARNING: the "{}" is a legacy replacer that will be removed in the future release. Please, consider '
              'using replacers defined in the "extensions/front/tf/ObjectDetectionAPI.py"'.format(self.replacement_id))
        log.debug('PostprocessorReplacement.generate_sub_graph')
        log.debug('matched_nodes = {}'.format(match.matched_nodes_names()))

        # softmax to be applied to the confidence
        softmax_conf_op = Softmax(graph, {'axis': 2, 'nchw_layout': True})
        softmax_conf_node = softmax_conf_op.add_node(dict(name='DetectionOutput_SoftMax_conf_'))

        # IE DetectionOutput layer consumes flattened tensors
        # reshape operation to flatten locations tensor
        reshape_loc_op = Reshape(graph, {'dim': np.array([0, -1])})
        reshape_loc_node = reshape_loc_op.add_node(dict(name='DetectionOutput_Reshape_loc_'))

        # IE DetectionOutput layer consumes flattened tensors
        # reshape operation to flatten confidence tensor
        reshape_conf_op = Reshape(graph, {'dim': np.array([0, -1])})
        reshape_conf_node = reshape_conf_op.add_node(dict(name='DetectionOutput_Reshape_conf_'))

        # create Node object from Op class
        detection_output_op = DetectionOutput(graph, match.custom_replacement_desc.custom_attributes)
        detection_output_op.attrs['old_infer'] = detection_output_op.attrs['infer']
        detection_output_op.attrs['infer'] = __class__.do_infer
        detection_output_node = detection_output_op.add_node(dict(name=detection_output_op.attrs['type'] + '_'))

        # create internal edges of the sub-graph. In this case we add edges to connect input port 0 and 1 of the
        # detection output with output of reshape of locations and reshape of confidence
        create_edge(softmax_conf_node, reshape_conf_node, 0, 0)
        create_edge(reshape_loc_node, detection_output_node, 0, 0)
        create_edge(reshape_conf_node, detection_output_node, 0, 1)
        return {'detection_output_node': detection_output_node, 'reshape_conf_node': softmax_conf_node,
                'reshape_loc_node': reshape_loc_node}

    @staticmethod
    def do_infer(node: Node):
        prior_boxes = node.in_node(2).value
        assert prior_boxes is not None
        # these are default variances values
        variance = np.array([[0.1, 0.1, 0.2, 0.2]])
        # replicating the variance values for all prior-boxes
        variances = np.tile(variance, [prior_boxes.shape[0], 1])
        # DetectionOutput in the Inference Engine expects the prior-boxes in the following layout: (values, variances)
        prior_boxes = np.concatenate((prior_boxes, variances), 0)
        # compared to the IE's DetectionOutput, the TF keeps the prior-boxes in YXYX, need to get back to the XYXY
        prior_boxes = np.concatenate((prior_boxes[:, 1:2], prior_boxes[:, 0:1],
                                      prior_boxes[:, 3:4], prior_boxes[:, 2:3]), 1)
        #  adding another dimensions, as the prior-boxes are expected as 3d tensors
        prior_boxes = prior_boxes.reshape((1, 2, -1))
        node.in_node(2).shape = np.array(prior_boxes.shape, dtype=np.int64)
        node.in_node(2).value = prior_boxes

        node.old_infer(node)
        # compared to the IE's DetectionOutput, the TF keeps the locations in YXYX, need to get back to the XYXY
        # for last convolutions that operate the locations need to swap the X and Y for output feature weights & biases
        conv_nodes = backward_bfs_for_operation(node.in_node(0), ['Conv2D'])
        swap_weights_xy(conv_nodes)
        squeeze_reshape_and_concat(conv_nodes)
