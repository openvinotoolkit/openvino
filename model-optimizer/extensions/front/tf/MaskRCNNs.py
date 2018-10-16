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
import networkx as nx
import numpy as np

from extensions.front.sub import Sub
from extensions.front.tf.Preprocessor import PreprocessorReplacement
from extensions.ops.proposal import ProposalOp
from extensions.ops.DetectionOutput import DetectionOutput
from mo.front.common.partial_infer.utils import mark_input_bins, assign_dims_to_weights
from mo.front.common.weights import swap_weights_xy
from mo.front.extractor import update_attrs, update_ie_fields
from mo.front.subgraph_matcher import SubgraphMatch
from mo.front.tf.replacement import FrontReplacementFromConfigFileSubGraph
from mo.graph.graph import insert_node_after, Node
from mo.ops.activation import Activation
from mo.ops.const import Const
from mo.ops.crop import Crop
from mo.ops.concat import Concat
from mo.ops.eltwise import Eltwise
from mo.ops.op import Op
from mo.ops.output import Output
from mo.ops.permute import Permute
from mo.ops.reshape import Reshape
from mo.ops.roipooling import ROIPooling
from mo.ops.softmax import Softmax
from mo.utils.graph import backward_bfs_for_operation, scope_output_nodes
from mo.utils.error import Error


class TFObjectDetectionAPIMaskRCNNProposalAndROIPooling(FrontReplacementFromConfigFileSubGraph):
    """
    This class replaces sub-graph of operations with Proposal and ROIPooling layers and additional layers transforming
    tensors from layout of TensorFlow to layout required by Inference Engine.
    Refer to comments inside the function for more information about performed actions.
    """
    replacement_id = 'TFObjectDetectionAPIMaskRCNNProposalAndROIPooling'

    def run_after(self):
        return [PreprocessorReplacement]

    def run_before(self):
        return [TFObjectDetectionAPIMaskRCNNROIPoolingSecond, Sub]

    def output_edges_match(self, graph: nx.DiGraph, match: SubgraphMatch, new_sub_graph: dict):
        # the DetectionOutput in IE produces single tensor, but in TF it produces two tensors, so we need to create only
        # one output edge match
        return {match.output_node(0)[0].id: new_sub_graph['roi_pooling_node'].id}

    def nodes_to_remove(self, graph: nx.MultiDiGraph, match: SubgraphMatch):
        new_list = match.matched_nodes_names().copy()
        # do not remove nodes that produce box predictions and class predictions
        new_list.remove(match.single_input_node(0)[0].id)
        new_list.remove(match.single_input_node(1)[0].id)
        return new_list

    def generate_sub_graph(self, graph: nx.MultiDiGraph, match: SubgraphMatch):
        print('WARNING: the "{}" is a legacy replacer that will be removed in the future release. Please, consider '
              'using replacers defined in the "extensions/front/tf/ObjectDetectionAPI.py"'.format(self.replacement_id))
        log.debug('TFObjectDetectionAPIMaskRCNNProposalAndROIPooling: matched_nodes = {}'.
                  format(match.matched_nodes_names()))

        config_attrs = match.custom_replacement_desc.custom_attributes
        nms_threshold = config_attrs['nms_threshold']
        input_height = config_attrs['input_height']
        input_width = config_attrs['input_width']
        feat_stride = config_attrs['feat_stride']
        max_proposals = config_attrs['max_proposals']
        anchor_base_size = config_attrs['anchor_base_size']
        roi_spatial_scale = config_attrs['roi_spatial_scale']
        proposal_ratios = config_attrs['anchor_aspect_ratios']
        proposal_scales = config_attrs['anchor_scales']
        anchors_count = len(proposal_ratios) * len(proposal_scales)

        # get the ROIPool size from the CropAndResize which performs the same action
        if 'CropAndResize' not in graph.nodes():
            raise Error('Failed to find node with name "CropAndResize" in the topology. Probably this is not Faster'
                        ' RCNN topology or it is not supported')
        roi_pool_size = Node(graph, 'CropAndResize').in_node(3).value[0]

        # Convolution/matmul node that produces classes predictions
        # Permute result of the tensor with classes permissions so it will be in a correct layout for Softmax
        predictions_node = match.single_input_node(1)[0].in_node(0).in_node(0)
        permute_predictions_op = Permute(graph, {'order': np.array([0, 2, 3, 1])})
        permute_predictions_node = permute_predictions_op.create_node([], dict(name=predictions_node.name + '/Permute_'))
        insert_node_after(predictions_node, permute_predictions_node, 0)

        # create constant input with the image height, width and scale H and scale W (if present) required for Proposal
        const_value = np.array([[input_height, input_width, 1]], dtype=np.float32)
        const_op = Const(graph, dict(value=const_value, shape=const_value.shape))
        const_node = const_op.create_node([], dict(name='Proposal_const_image_size_'))

        reshape_classes_op = Reshape(graph, {'dim': np.array([0, -1, 2])})
        reshape_classes_node = reshape_classes_op.create_node([permute_predictions_node],
                                                              dict(name='Reshape_FirstStageBoxPredictor_Class_'))
        update_attrs(reshape_classes_node, 'shape_attrs', 'dim')

        softmax_conf_op = Softmax(graph, {'axis': 1})
        softmax_conf_node = softmax_conf_op.create_node([reshape_classes_node],
                                                        dict(name='FirstStageBoxPredictor_SoftMax_Class_'))

        reshape_softmax_op = Reshape(graph, {'dim': np.array([1, anchors_count, 2, -1])})
        reshape_softmax_node = reshape_softmax_op.create_node([softmax_conf_node], dict(name='Reshape_Softmax_Class_'))
        update_attrs(reshape_softmax_node, 'shape_attrs', 'dim')

        permute_reshape_softmax_op = Permute(graph, {'order': np.array([0, 1, 3, 2])})
        permute_reshape_softmax_node = permute_reshape_softmax_op.create_node([reshape_softmax_node],
                                                                              dict(name='Permute_'))

        # implement custom reshape infer function because we need to know the input convolution node output dimension
        # sizes but we can know it only after partial infer
        reshape_permute_op = Reshape(graph, {'dim': np.ones([4]), 'anchors_count': anchors_count,
                                             'conv_node': predictions_node})
        reshape_permute_op.attrs['old_infer'] = reshape_permute_op.attrs['infer']
        reshape_permute_op.attrs['infer'] = __class__.classes_probabilities_reshape_shape_infer
        reshape_permute_node = reshape_permute_op.create_node([permute_reshape_softmax_node],
                                                              dict(name='Reshape_Permute_Class_'))
        update_attrs(reshape_permute_node, 'shape_attrs', 'dim')

        proposal_op = ProposalOp(graph, dict(min_size=10, framework='tensorflow', box_coordinate_scale=10,
                                             box_size_scale=5, post_nms_topn=max_proposals, feat_stride=feat_stride,
                                             ratio=proposal_ratios, scale=proposal_scales, base_size=anchor_base_size,
                                             pre_nms_topn=2**31 - 1,
                                             nms_thresh=nms_threshold))
        proposal_node = proposal_op.create_node([reshape_permute_node,
                                                 match.single_input_node(0)[0].in_node(0).in_node(0),
                                                 const_node],
                                                dict(name=proposal_op.attrs['type'] + '_'))

        proposal_reshape_4d_op = Reshape(graph, {'dim': np.array([max_proposals, 1, 1, 5])})
        proposal_reshape_4d_node = proposal_reshape_4d_op.create_node([proposal_node], dict(name="reshape_4d_"))
        update_attrs(proposal_reshape_4d_node, 'shape_attrs', 'dim')

        # create convolution node to swap X and Y coordinates in the proposals
        conv_filter_const_data = np.array(np.array([[1, 0, 0, 0, 0],
                                                    [0, 0, 1, 0, 0],
                                                    [0, 1, 0, 0, 0],
                                                    [0, 0, 0, 0, 1],
                                                    [0, 0, 0, 1, 0]],
                                                   dtype=np.float32).reshape([1, 1, 5, 5]), dtype=np.float32)
        conv_filter_const_op = Const(graph, dict(value=conv_filter_const_data, spatial_dims=np.array([2, 3])))
        conv_filter_const_node = conv_filter_const_op.create_node([], dict(name="conv_weights"))

        conv_op = Op(graph, {
                        'op': 'Conv2D',
                        'bias_addable': False,
                        'spatial_dims': np.array([1, 2]),
                        'channel_dims': np.array([3]),
                        'batch_dims': np.array([0]),
                        'pad': None,
                        'pad_spatial_shape': None,
                        'input_feature_channel': [2],
                        'output_feature_channel': [2],
                        'output_shape': [max_proposals, 1, 1, 5],
                        'dilation': np.array([1, 1, 1, 1], dtype=np.int64),
                        'stride': np.array([1, 1, 1, 1]),
                        'type': 'Convolution',
                        'group': None,
                        'layout': 'NHWC',
                        'infer': __class__.fake_conv_shape_infer})
        predictions_node = conv_op.create_node([proposal_reshape_4d_node, conv_filter_const_node], dict(name="conv_"))
        update_ie_fields(graph.node[predictions_node.id])

        proposal_reshape_2d_op = Reshape(graph, {'dim': np.array([max_proposals, 5])})
        proposal_reshape_2d_node = proposal_reshape_2d_op.create_node([predictions_node], dict(name="reshape_2d_"))
        # set specific name for this Reshape operation so we can use it in the DetectionOutput replacer
        proposal_reshape_2d_node['name'] = 'swapped_proposals'

        # the TF implementation of Proposal with bi-linear filtration need proposals scaled by image size
        proposal_scale_const = np.array([1.0, 1 / input_height, 1 / input_width, 1 / input_height, 1 / input_width],
                                        dtype=np.float32)
        proposal_scale_const_op = Const(graph, dict(value=proposal_scale_const, shape=proposal_scale_const.shape))
        proposal_scale_const_node = proposal_scale_const_op.create_node([], dict(name='Proposal_scale_const_'))

        scale_proposals_op = Eltwise(graph, {'operation': 'mul'})
        scale_proposals_node = scale_proposals_op.create_node([proposal_reshape_2d_node, proposal_scale_const_node],
                                                              dict(name='scale_proposals_'))

        feature_extractor_output_nodes = scope_output_nodes(graph, 'FirstStageFeatureExtractor')
        if len(feature_extractor_output_nodes) != 1:
            raise Error("Failed to determine FirstStageFeatureExtractor output node to connect it to the ROIPooling."
                        "Found the following nodes: {}".format([node.name for node in feature_extractor_output_nodes]))

        roi_pooling_op = ROIPooling(graph, dict(method="bilinear", framework="tensorflow",
                                                pooled_h=roi_pool_size, pooled_w=roi_pool_size,
                                                spatial_scale=roi_spatial_scale))
        roi_pooling_node = roi_pooling_op.create_node([feature_extractor_output_nodes[0], scale_proposals_node],
                                                      dict(name='ROI_Pooling_'))

        return {'roi_pooling_node': roi_pooling_node}

    @staticmethod
    def fake_conv_shape_infer(node: Node):
        node.out_node(0).shape = node.in_node(0).shape
        # call functions to update internal attributes required for correct IR generation
        mark_input_bins(node)
        assign_dims_to_weights(node.in_node(1), [0, 1], node.input_feature_channel, node.output_feature_channel, 4)

    @staticmethod
    def classes_probabilities_reshape_shape_infer(node: Node):
        # now we can determine the reshape dimensions from Convolution node
        conv_node = node.conv_node
        conv_output_shape = conv_node.out_node().shape

        # update desired shape of the Reshape node
        node.dim = np.array([0, conv_output_shape[1], conv_output_shape[2], node.anchors_count * 2])
        node.old_infer(node)


class TFObjectDetectionAPIMaskRCNNDetectionOutput(FrontReplacementFromConfigFileSubGraph):
    replacement_id = 'TFObjectDetectionAPIMaskRCNNDetectionOutput'

    def run_before(self):
        return [TFObjectDetectionAPIMaskRCNNROIPoolingSecond]

    def run_after(self):
        return [TFObjectDetectionAPIMaskRCNNProposalAndROIPooling]

    def nodes_to_remove(self, graph: nx.MultiDiGraph, match: SubgraphMatch):
        new_nodes_to_remove = match.matched_nodes_names()
        new_nodes_to_remove.extend(['detection_boxes', 'detection_scores', 'num_detections', 'Reshape_16'])
        return new_nodes_to_remove

    def output_edges_match(self, graph: nx.DiGraph, match: SubgraphMatch, new_sub_graph: dict):
        # the DetectionOutput in IE produces single tensor, but in TF it produces two tensors, so we need to create only
        # one output edge match
        return {match.output_node(0)[0].id: new_sub_graph['detection_output_node'].id}

    def generate_sub_graph(self, graph: nx.MultiDiGraph, match: SubgraphMatch):
        print('WARNING: the "{}" is a legacy replacer that will be removed in the future release. Please, consider '
              'using replacers defined in the "extensions/front/tf/ObjectDetectionAPI.py"'.format(self.replacement_id))
        log.debug('TFObjectDetectionAPIMaskRCNNDetectionOutput: matched_nodes = {}'.format(match.matched_nodes_names()))

        config_attrs = match.custom_replacement_desc.custom_attributes
        num_classes = config_attrs['num_classes']
        max_detections_per_class = config_attrs['max_detections_per_class']

        # softmax to be applied to the confidence
        softmax_conf_op = Softmax(graph, {'axis': 1, 'nchw_layout': True})
        softmax_conf_node = softmax_conf_op.create_node([match.single_input_node(1)[0].in_node(0)],
                                                        dict(name='DetectionOutput_SoftMax_conf_'))

        # IE DetectionOutput layer consumes flattened tensors
        # reshape operation to flatten confidence tensor
        reshape_conf_op = Reshape(graph, {'dim': np.array([1, -1])})
        reshape_conf_node = reshape_conf_op.create_node([softmax_conf_node], dict(name='DetectionOutput_Reshape_conf_'))

        # TF produces locations tensor without boxes for background.
        # Inference Engine DetectionOutput layer requires background boxes so we generate them with some values
        # and concatenate with locations tensor
        fake_background_locs_blob = np.tile([[[1, 1, 2, 2]]], [max_detections_per_class, 1, 1])
        fake_background_locs_const_op = Const(graph, dict(value=fake_background_locs_blob,
                                                          shape=fake_background_locs_blob.shape))
        fake_background_locs_const_node = fake_background_locs_const_op.create_node([])

        reshape_loc_op = Reshape(graph, {'dim': np.array([max_detections_per_class, num_classes, 4])})
        reshape_loc_node = reshape_loc_op.create_node([match.single_input_node(0)[0].in_node(0)],
                                                      dict(name='Reshape_loc_'))

        concat_loc_op = Concat(graph, {'axis': 1})
        concat_loc_node = concat_loc_op.create_node([fake_background_locs_const_node, reshape_loc_node],
                                                    dict(name='Concat_fake_loc_'))

        # blob with variances
        variances_blob = np.array([0.1, 0.1, 0.2, 0.2])
        variances_const_op = Const(graph, dict(value=variances_blob, shape=variances_blob.shape))
        variances_const_node = variances_const_op.create_node([])

        # reshape locations tensor to 2D so it could be passed to Eltwise which will be converted to ScaleShift
        reshape_loc_2d_op = Reshape(graph, {'dim': np.array([-1, 4])})
        reshape_loc_2d_node = reshape_loc_2d_op.create_node([concat_loc_node], dict(name='reshape_locs_2d_'))

        # element-wise multiply locations with variances
        eltwise_locs_op = Eltwise(graph, {'operation': 'mul'})
        eltwise_locs_node = eltwise_locs_op.create_node([reshape_loc_2d_node, variances_const_node],
                                                        dict(name='scale_locs_'))

        # IE DetectionOutput layer consumes flattened tensors
        reshape_loc_do_op = Reshape(graph, {'dim': np.array([1, -1])})
        reshape_loc_do_node = reshape_loc_do_op.create_node([eltwise_locs_node],
                                                            dict(name='DetectionOutput_reshape_locs_'))

        # find Proposal output with swapped X and Y coordinates
        proposal_nodes_ids = [node_id for node_id, attrs in graph.nodes(data=True)
                              if 'name' in attrs and attrs['name'] == 'swapped_proposals']
        if len(proposal_nodes_ids) != 1:
            raise Error("Found the following nodes '{}' with name 'swapped_proposals' but there should be exactly 1.".
                        format(proposal_nodes_ids))
        priors_node = Node(graph, proposal_nodes_ids[0])

        # reshape priors boxes as Detection Output expects
        reshape_priors_op = Reshape(graph, {'dim': np.array([1, 1, -1])})
        reshape_priors_node = reshape_priors_op.create_node([priors_node], dict(name='DetectionOutput_reshape_priors_'))

        detection_output_op = DetectionOutput(graph, match.custom_replacement_desc.custom_attributes)
        # update infer function to re-pack weights
        detection_output_op.attrs['old_infer'] = detection_output_op.attrs['infer']
        detection_output_op.attrs['infer'] = __class__.do_infer
        detection_output_node = detection_output_op.create_node(
            [reshape_loc_do_node, reshape_conf_node, reshape_priors_node],
            dict(name=detection_output_op.attrs['type'] + '_', share_location=0, normalized=0,
                 variance_encoded_in_target=1))
        # set specific name to the node so we can find it in the next replacer
        detection_output_node.name = 'detection_output'

        return {'detection_output_node': detection_output_node}

    @staticmethod
    def do_infer(node):
        node.old_infer(node)
        # compared to the IE's DetectionOutput, the TF keeps the locations in YXYX, need to get back to the XYXY
        # for last matmul that operate the locations need to swap the X and Y for output feature weights & biases
        swap_weights_xy(backward_bfs_for_operation(node.in_node(0), ['MatMul']))


class TFObjectDetectionAPIMaskRCNNROIPoolingSecond(FrontReplacementFromConfigFileSubGraph):
    replacement_id = 'TFObjectDetectionAPIMaskRCNNROIPoolingSecond'

    def output_edges_match(self, graph: nx.DiGraph, match: SubgraphMatch, new_sub_graph: dict):
        # the DetectionOutput in IE produces single tensor, but in TF it produces two tensors, so we need to create only
        # one output edge match
        return {match.output_node(0)[0].id: new_sub_graph['roi_pooling_node'].id}

    def generate_sub_graph(self, graph: nx.MultiDiGraph, match: SubgraphMatch):
        print('WARNING: the "{}" is a legacy replacer that will be removed in the future release. Please, consider '
              'using replacers defined in the "extensions/front/tf/ObjectDetectionAPI.py"'.format(self.replacement_id))
        log.debug('TFObjectDetectionAPIMaskRCNNROIPoolingSecond: matched_nodes = {}'.
                  format('\n'.join(match.matched_nodes_names())))

        config_attrs = match.custom_replacement_desc.custom_attributes
        roi_spatial_scale = config_attrs['roi_spatial_scale']

        # get the ROIPool size from the CropAndResize which performs the same action
        if 'CropAndResize_1' not in graph.nodes():
            raise Error('Failed to find node with name "CropAndResize_1" in the topology. Probably this is not Mask'
                        ' RCNN topology or it is not supported')
        roi_pool_size = Node(graph, 'CropAndResize_1').in_node(3).value[0]

        detection_output_nodes_ids = [node_id for node_id, attrs in graph.nodes(data=True)
                                      if 'name' in attrs and attrs['name'] == 'detection_output']
        if len(detection_output_nodes_ids) != 1:
            raise Error("Found the following nodes '{}' with type 'DetectionOutput' but there should be exactly 1.".
                        format(detection_output_nodes_ids))
        detection_output_node = Node(graph, detection_output_nodes_ids[0])

        # create crop blob that gets input from the DetectionOutput layer and gets tensor with classes probabilities
        # and bounding boxes to feed it to ROIPooling
        crop_shape_const_blob = np.ones(shape=[1, 1, 100, 5])
        crop_shape_const_op = Const(graph, dict(value=crop_shape_const_blob))
        crop_shape_const_node = crop_shape_const_op.create_node([])

        crop_op = Crop(graph, dict(axis=np.array([3]), offset=np.array([2]), dim=np.array([5]), nchw_layout=True))
        crop_node = crop_op.create_node([detection_output_node, crop_shape_const_node], dict(name='Crop_DO_'))

        # reshape bounding boxes as required by ROIPooling
        reshape_DO_op = Reshape(graph, dict(dim=np.array([-1, 5])))
        reshape_DO_node = reshape_DO_op.create_node([crop_node], dict(name='Reshape_DO_'))

        roi_pooling_op = ROIPooling(graph, dict(method="bilinear", framework="tensorflow",
                                                pooled_h=roi_pool_size, pooled_w=roi_pool_size,
                                                spatial_scale=roi_spatial_scale))
        roi_pooling_node = roi_pooling_op.create_node([match.single_input_node(0)[0].in_node(), reshape_DO_node],
                                                      dict(name='ROI_Pooling_'))
        return {'roi_pooling_node': roi_pooling_node}


class TFObjectDetectionAPIMaskRCNNSigmoid(FrontReplacementFromConfigFileSubGraph):
    """
    Adds activation with sigmoid function to the end of the network producing masks tensors.
    """
    replacement_id = 'TFObjectDetectionAPIMaskRCNNSigmoid'

    def run_after(self):
        return [TFObjectDetectionAPIMaskRCNNROIPoolingSecond]

    def nodes_to_remove(self, graph: nx.MultiDiGraph, match: SubgraphMatch):
        # do not remove matched node
        return []

    def output_edges_match(self, graph: nx.DiGraph, match: SubgraphMatch, new_sub_graph: dict):
        # the DetectionOutput in IE produces single tensor, but in TF it produces two tensors, so we need to create only
        # one output edge match
        return {match.output_node(0)[0].id: new_sub_graph['output_node'].id}

    def generate_sub_graph(self, graph: nx.MultiDiGraph, match: SubgraphMatch):
        print('WARNING: the "{}" is a legacy replacer that will be removed in the future release. Please, consider '
              'using replacers defined in the "extensions/front/tf/ObjectDetectionAPI.py"'.format(self.replacement_id))
        log.debug('TFObjectDetectionAPIMaskRCNNSigmoid: matched_nodes = {}'.
                  format('\n'.join(match.matched_nodes_names())))

        input_node = match.single_input_node(0)[0]
        sigmoid_op = Activation(graph, dict(operation='sigmoid'))
        sigmoid_node = sigmoid_op.create_node([input_node], dict(name=input_node.id + '/Sigmoid_'))
        sigmoid_node.name = 'masks'

        # add special node of type "Output" that is a marker for the output nodes of the topology
        output_op = Output(graph, dict(name="Sigmoid_OutputOp"))
        output_node = output_op.create_node([sigmoid_node])

        print('The support of mask R-CNN topologies is in a preview mode.\n'
              'The graph output nodes "num_detections", "detection_boxes", "detection_classes", "detection_scores", '
              '"detection_masks" have been replaced with two layers: "masks" containing predicted instance masks for '
              'each bounding box generated with a "detection_output" layer.\n'
              'Refer to IR catalogue in the Inference Engine documentation for information about the DetectionOutput '
              'layer and Inference Engine documentation about output data interpretation.')

        return {'output_node': output_node}
