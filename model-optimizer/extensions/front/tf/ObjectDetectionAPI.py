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

import collections
import logging as log

import networkx as nx
import numpy as np

from extensions.front.standalone_const_eraser import StandaloneConstEraser
from extensions.front.sub import Sub
from extensions.front.tf.Pack import Pack
from extensions.ops.DetectionOutput import DetectionOutput
from extensions.ops.proposal import ProposalOp
from mo.front.common.partial_infer.utils import mark_input_bins, assign_dims_to_weights
from mo.front.common.weights import swap_weights_xy
from mo.front.extractor import add_output_ops, update_attrs, update_ie_fields
from mo.front.extractor import output_user_data_repack
from mo.front.subgraph_matcher import SubgraphMatch
from mo.front.tf.replacement import FrontReplacementFromConfigFileSubGraph, FrontReplacementFromConfigFileGeneral
from mo.graph.graph import create_edge, insert_node_after, Node
from mo.ops.activation import Activation
from mo.ops.concat import Concat
from mo.ops.const import Const
from mo.ops.crop import Crop
from mo.ops.div import Div
from mo.ops.eltwise import Eltwise
from mo.ops.expand_dims import ExpandDims
from mo.ops.op import Op
from mo.ops.output import Output
from mo.ops.permute import Permute
from mo.ops.reshape import Reshape
from mo.ops.roipooling import ROIPooling
from mo.ops.softmax import Softmax
from mo.utils.error import Error
from mo.utils.graph import backward_bfs_for_operation, scope_output_nodes
from mo.utils.pipeline_config import PipelineConfig

missing_param_error = 'To convert the model specify path to the pipeline configuration file which was used to ' \
                      'generate the model. Please use "--tensorflow_object_detection_api_pipeline_config" option:\n' \
                      '--tensorflow_object_detection_api_pipeline_config "<path_to_pipeline.config>"\nIf you have ' \
                      'downloaded the model file from the Object Detection Model zoo repository then this file is ' \
                      'located in the archive with frozen model and called "pipeline.config".\nIf you did not use ' \
                      'this command line parameter before that means that you are using currently deprecated ' \
                      'TensorFlow* Object Detection API models conversion mechanism. If you still want to use this ' \
                      'mechanism then specify legacy configuration file: legacy_ssd_support.json, ' \
                      'legacy_ssd_v2_support.json, legacy_faster_rcnn.json or legacy_mask_rcnn.json depending on ' \
                      'your topology. Note, that models converted with a deprecated mechanism:\n' \
                      '1. Produce worse results than models converted with a new one.\n' \
                      '2. Require manual change of the configuration file (for specific topologies).\n' \
                      '3. Have longer command line parameters list.\nRefer to the Model Optimizer documentation for ' \
                      'more information how to use new mechanism.'


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
    q = collections.deque()
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


class ObjectDetectionAPIPreprocessorReplacement(FrontReplacementFromConfigFileSubGraph):
    """
    The class replaces the "Preprocessor" block resizing input image and applying mean/scale values. Only nodes related
    to applying mean/scaling values are kept.
    """
    replacement_id = 'ObjectDetectionAPIPreprocessorReplacement'

    def run_before(self):
        return [Pack, Sub]

    def nodes_to_remove(self, graph: nx.MultiDiGraph, match: SubgraphMatch):
        new_nodes_to_remove = match.matched_nodes_names()
        # do not remove nodes that perform input image scaling and mean value subtraction
        for node_to_keep in ('Preprocessor/sub', 'Preprocessor/sub/y', 'Preprocessor/mul', 'Preprocessor/mul/x'):
            if node_to_keep in new_nodes_to_remove:
                new_nodes_to_remove.remove(node_to_keep)
        return new_nodes_to_remove

    def generate_sub_graph(self, graph: nx.MultiDiGraph, match: SubgraphMatch):
        argv = graph.graph['cmd_params']
        if argv.tensorflow_object_detection_api_pipeline_config is None:
            raise Error(missing_param_error)
        pipeline_config = PipelineConfig(argv.tensorflow_object_detection_api_pipeline_config)

        sub_node = match.output_node(0)[0]
        if not sub_node.has('op') or sub_node.op != 'Sub':
            raise Error('The output op of the Preprocessor sub-graph is not of type "Sub". Looks like the topology is '
                        'not created with TensorFlow Object Detection API.')

        mul_node = None
        if sub_node.in_node(0).has('op') and sub_node.in_node(0).op == 'Mul':
            log.info('There is image scaling node in the Preprocessor block.')
            mul_node = sub_node.in_node(0)

        initial_input_node_name = 'image_tensor'
        if initial_input_node_name not in graph.nodes():
            raise Error('Input node "{}" of the graph is not found. Do not run the Model Optimizer with '
                        '"--input" command line parameter.'.format(initial_input_node_name))
        placeholder_node = Node(graph, initial_input_node_name)

        placeholder_node.shape[0] = 1  # batch size 1 by default. It will be overridden if user specifies "-b" option
        placeholder_node.shape[1] = pipeline_config.get_param('preprocessed_image_height')
        placeholder_node.shape[2] = pipeline_config.get_param('preprocessed_image_width')

        to_float_node = placeholder_node.out_node(0)
        if not to_float_node.has('op') or to_float_node.op != 'Cast':
            raise Error('The output of the node "{}" is not Cast operation. Cannot apply replacer.'.format(
                initial_input_node_name))

        # connect to_float_node directly with node performing scale on mean value subtraction
        if mul_node is None:
            create_edge(to_float_node, sub_node, 0, 0)
        else:
            create_edge(to_float_node, mul_node, 0, 1)

        print('The Preprocessor block has been removed. Only nodes performing mean value subtraction and scaling (if'
              ' applicable) are kept.')
        return {}


class ObjectDetectionAPIDetectionOutputReplacement(FrontReplacementFromConfigFileSubGraph):
    """
    Replaces sub-graph that is equal to the DetectionOutput layer from InferenceEngine. This replacer is used for Faster
    R-CNN and Mask R-CNN topologies conversion.
    """
    replacement_id = 'ObjectDetectionAPIDetectionOutputReplacement'

    def run_before(self):
        return [ObjectDetectionAPIMaskRCNNROIPoolingSecondReplacement]

    def run_after(self):
        return [ObjectDetectionAPIProposalAndROIPoolingReplacement]

    def nodes_to_remove(self, graph: nx.MultiDiGraph, match: SubgraphMatch):
        new_nodes_to_remove = match.matched_nodes_names().copy()
        new_nodes_to_remove.extend(['detection_boxes', 'detection_scores', 'num_detections'])
        return new_nodes_to_remove

    def output_edges_match(self, graph: nx.DiGraph, match: SubgraphMatch, new_sub_graph: dict):
        # the DetectionOutput in IE produces single tensor, but in TF it produces four tensors, so we need to create
        # only one output edge match
        return {match.output_node(0)[0].id: new_sub_graph['detection_output_node'].id}

    def generate_sub_graph(self, graph: nx.MultiDiGraph, match: SubgraphMatch):
        argv = graph.graph['cmd_params']
        if argv.tensorflow_object_detection_api_pipeline_config is None:
            raise Error(missing_param_error)
        pipeline_config = PipelineConfig(argv.tensorflow_object_detection_api_pipeline_config)

        num_classes = pipeline_config.get_param('num_classes')
        first_stage_max_proposals = pipeline_config.get_param('first_stage_max_proposals')
        post_processing_function = pipeline_config.get_param('postprocessing_score_converter')

        if post_processing_function == 'SOFTMAX':
            # softmax to be applied to the confidence
            softmax_conf_op = Softmax(graph, dict(axis=1, nchw_layout=True))
            activation_conf_node = softmax_conf_op.create_node([match.single_input_node(1)[0].in_node(0)],
                                                               dict(name='do_softmax_conf'))
        elif post_processing_function == 'SIGMOID':
            # sigmoid activation function to be applied to the confidence
            sigmoid_conf_op = Activation(graph, dict(operation='sigmoid'))
            activation_conf_node = sigmoid_conf_op.create_node([match.single_input_node(1)[0].in_node(0)],
                                                               dict(name='do_sigmoid_conf'))
        elif post_processing_function == 'IDENTITY':
            # in case of Identity do nothing and just use result from the input node
            activation_conf_node = match.single_input_node(1)[0].in_node(0)
        else:
            raise Error('Unknown post-processing activation function "{}".'.format(post_processing_function))

        # IE DetectionOutput layer consumes flattened tensors
        # reshape operation to flatten confidence tensor
        reshape_conf_op = Reshape(graph, dict(dim=np.array([1, -1])))
        reshape_conf_node = reshape_conf_op.create_node([activation_conf_node], dict(name='do_reshape_conf'))

        # TF produces locations tensor without boxes for background.
        # Inference Engine DetectionOutput layer requires background boxes so we generate them with some values
        # and concatenate with locations tensor
        fake_background_locs_blob = np.tile([[[1, 1, 2, 2]]], [first_stage_max_proposals, 1, 1])
        fake_background_locs_const_op = Const(graph, dict(value=fake_background_locs_blob))
        fake_background_locs_const_node = fake_background_locs_const_op.create_node([])

        reshape_loc_op = Reshape(graph, dict(dim=np.array([first_stage_max_proposals, num_classes, 4])))
        reshape_loc_node = reshape_loc_op.create_node([match.single_input_node(0)[0].in_node(0)],
                                                      dict(name='reshape_loc'))

        concat_loc_op = Concat(graph, dict(axis=1))
        concat_loc_node = concat_loc_op.create_node([fake_background_locs_const_node, reshape_loc_node],
                                                    dict(name='concat_fake_loc'))

        # constant node with variances
        variances_const_op = Const(graph, dict(value=np.array([0.1, 0.1, 0.2, 0.2])))
        variances_const_node = variances_const_op.create_node([])

        # reshape locations tensor to 2D so it could be passed to Eltwise which will be converted to ScaleShift
        reshape_loc_2d_op = Reshape(graph, dict(dim=np.array([-1, 4])))
        reshape_loc_2d_node = reshape_loc_2d_op.create_node([concat_loc_node], dict(name='reshape_locs_2'))

        # element-wise multiply locations with variances
        eltwise_locs_op = Eltwise(graph, dict(operation='mul'))
        eltwise_locs_node = eltwise_locs_op.create_node([reshape_loc_2d_node, variances_const_node],
                                                        dict(name='scale_locs'))

        # IE DetectionOutput layer consumes flattened tensors
        reshape_loc_do_op = Reshape(graph, dict(dim=np.array([1, -1])))
        reshape_loc_do_node = reshape_loc_do_op.create_node([eltwise_locs_node], dict(name='do_reshape_locs'))

        # find Proposal output with swapped X and Y coordinates
        proposal_nodes_ids = [node_id for node_id, attrs in graph.nodes(data=True)
                              if 'name' in attrs and attrs['name'] == 'swapped_proposals']
        if len(proposal_nodes_ids) != 1:
            raise Error("Found the following nodes '{}' with name 'swapped_proposals' but there should be exactly 1. "
                        "Looks like ObjectDetectionAPIProposalAndROIPoolingReplacement replacement didn't work.".
                        format(proposal_nodes_ids))
        priors_node = Node(graph, proposal_nodes_ids[0])

        # reshape priors boxes as Detection Output expects
        reshape_priors_op = Reshape(graph, dict(dim=np.array([1, 1, -1])))
        reshape_priors_node = reshape_priors_op.create_node([priors_node], dict(name='DetectionOutput_reshape_priors_'))

        detection_output_op = DetectionOutput(graph, match.custom_replacement_desc.custom_attributes)
        # update infer function to re-pack weights
        detection_output_op.attrs['old_infer'] = detection_output_op.attrs['infer']
        detection_output_op.attrs['infer'] = __class__.do_infer
        detection_output_node = detection_output_op.create_node(
            [reshape_loc_do_node, reshape_conf_node, reshape_priors_node],
            dict(name=detection_output_op.attrs['type'], share_location=0, normalized=0, variance_encoded_in_target=1,
                 clip=1,
                 num_classes=num_classes,
                 input_height=pipeline_config.get_param('preprocessed_image_height'),
                 input_width=pipeline_config.get_param('preprocessed_image_width'),
                 confidence_threshold=pipeline_config.get_param('postprocessing_score_threshold'),
                 top_k=pipeline_config.get_param('postprocessing_max_detections_per_class'),
                 keep_top_k=pipeline_config.get_param('postprocessing_max_total_detections'),
                 nms_threshold=pipeline_config.get_param('postprocessing_iou_threshold')))
        # set specific name to the node so we can find it in other replacers
        detection_output_node.name = 'detection_output'

        # add special node of type "Output" that is a marker for the output nodes of the topology
        output_op = Output(graph, dict(name='do_OutputOp'))
        output_op.create_node([detection_output_node])

        print('The graph output nodes "num_detections", "detection_boxes", "detection_classes", "detection_scores" '
              'have been replaced with a single layer of type "Detection Output". Refer to IR catalogue in the '
              'Inference Engine documentation for information about this layer.')

        return {'detection_output_node': detection_output_node}

    @staticmethod
    def do_infer(node):
        node.old_infer(node)
        # compared to the IE's DetectionOutput, the TF keeps the locations in YXYX, need to get back to the XYXY
        # for last matmul that operate the locations need to swap the X and Y for output feature weights & biases
        swap_weights_xy(backward_bfs_for_operation(node.in_node(0), ['MatMul']))


class ObjectDetectionAPIMaskRCNNROIPoolingSecondReplacement(FrontReplacementFromConfigFileSubGraph):
    replacement_id = 'ObjectDetectionAPIMaskRCNNROIPoolingSecondReplacement'

    def output_edges_match(self, graph: nx.DiGraph, match: SubgraphMatch, new_sub_graph: dict):
        return {match.output_node(0)[0].id: new_sub_graph['roi_pooling_node'].id}

    def generate_sub_graph(self, graph: nx.MultiDiGraph, match: SubgraphMatch):
        argv = graph.graph['cmd_params']
        if argv.tensorflow_object_detection_api_pipeline_config is None:
            raise Error(missing_param_error)
        pipeline_config = PipelineConfig(argv.tensorflow_object_detection_api_pipeline_config)
        roi_pool_size = pipeline_config.get_param('initial_crop_size')

        detection_output_nodes_ids = [node_id for node_id, attrs in graph.nodes(data=True)
                                      if 'name' in attrs and attrs['name'] == 'detection_output']
        if len(detection_output_nodes_ids) != 1:
            raise Error("Found the following nodes '{}' with 'detection_output' but there should be exactly 1.".
                        format(detection_output_nodes_ids))
        detection_output_node = Node(graph, detection_output_nodes_ids[0])

        # create crop blob that gets input from the DetectionOutput layer and gets tensor with classes probabilities
        # and bounding boxes to feed it to ROIPooling
        crop_shape_const_blob = np.ones(
            shape=[1, 1, pipeline_config.get_param('postprocessing_max_total_detections'), 5])
        crop_shape_const_op = Const(graph, dict(value=crop_shape_const_blob, nchw_layout=True))
        crop_shape_const_node = crop_shape_const_op.create_node([])

        crop_op = Crop(graph, dict(axis=np.array([3]), offset=np.array([2]), dim=np.array([5]), nchw_layout=True))
        crop_node = crop_op.create_node([detection_output_node, crop_shape_const_node], dict(name='crop_do'))

        # reshape bounding boxes as required by ROIPooling
        reshape_do_op = Reshape(graph, dict(dim=np.array([-1, 5])))
        reshape_do_node = reshape_do_op.create_node([crop_node], dict(name='reshape_do'))

        roi_pooling_op = ROIPooling(graph, dict(method="bilinear", framework="tensorflow", spatial_scale=1,
                                                pooled_h=roi_pool_size, pooled_w=roi_pool_size))
        roi_pooling_node = roi_pooling_op.create_node([match.single_input_node(0)[0].in_node(), reshape_do_node],
                                                      dict(name='ROI_pooling_2'))
        return {'roi_pooling_node': roi_pooling_node}


class ObjectDetectionAPIMaskRCNNSigmoidReplacement(FrontReplacementFromConfigFileGeneral):
    """
    This replacer is used to convert Mask R-CNN topologies only.
    Adds activation with sigmoid function to the end of the network producing masks tensors.
    """
    replacement_id = 'ObjectDetectionAPIMaskRCNNSigmoidReplacement'

    def run_after(self):
        return [ObjectDetectionAPIMaskRCNNROIPoolingSecondReplacement]

    def transform_graph(self, graph: nx.MultiDiGraph, replacement_descriptions):
        output_node = None
        op_outputs = [n for n, d in graph.nodes(data=True) if 'op' in d and d['op'] == 'OpOutput']
        for op_output in op_outputs:
            last_node = Node(graph, op_output).in_node(0)
            if last_node.name.startswith('SecondStageBoxPredictor'):
                sigmoid_op = Activation(graph, dict(operation='sigmoid'))
                sigmoid_node = sigmoid_op.create_node([last_node], dict(name=last_node.id + '/sigmoid'))
                sigmoid_node.name = 'masks'

                if output_node is not None:
                    raise Error('Identified two possible outputs from the topology. Cannot proceed.')
                # add special node of type "Output" that is a marker for the output nodes of the topology
                output_op = Output(graph, dict(name=sigmoid_node.name + '/OutputOp'))
                output_node = output_op.create_node([sigmoid_node])

        print('The predicted masks are saved in the "masks" layer for each bounding box generated with a '
              '"detection_output" layer.\n Refer to IR catalogue in the Inference Engine documentation for information '
              'about the DetectionOutput layer and Inference Engine documentation about output data interpretation.\n'
              'The topology can be inferred using dedicated sample "mask_rcnn_sample".')


class ObjectDetectionAPIProposalAndROIPoolingReplacement(FrontReplacementFromConfigFileSubGraph):
    """
    This class replaces sub-graph of operations with Proposal and ROIPooling layers and additional layers transforming
    tensors from layout of TensorFlow to layout required by Inference Engine.
    Refer to comments inside the function for more information about performed actions.
    """
    replacement_id = 'ObjectDetectionAPIProposalAndROIPoolingReplacement'

    def run_after(self):
        return [ObjectDetectionAPIPreprocessorReplacement]

    def run_before(self):
        return [Sub]

    def output_edges_match(self, graph: nx.DiGraph, match: SubgraphMatch, new_sub_graph: dict):
        return {match.output_node(0)[0].id: new_sub_graph['roi_pooling_node'].id}

    def nodes_to_remove(self, graph: nx.MultiDiGraph, match: SubgraphMatch):
        new_list = match.matched_nodes_names().copy()
        # do not remove nodes that produce box predictions and class predictions
        new_list.remove(match.single_input_node(0)[0].id)
        new_list.remove(match.single_input_node(1)[0].id)
        return new_list

    def generate_sub_graph(self, graph: nx.MultiDiGraph, match: SubgraphMatch):
        argv = graph.graph['cmd_params']
        if argv.tensorflow_object_detection_api_pipeline_config is None:
            raise Error(missing_param_error)
        pipeline_config = PipelineConfig(argv.tensorflow_object_detection_api_pipeline_config)

        input_height = pipeline_config.get_param('preprocessed_image_height')
        input_width = pipeline_config.get_param('preprocessed_image_width')
        max_proposals = pipeline_config.get_param('first_stage_max_proposals')
        roi_pool_size = pipeline_config.get_param('initial_crop_size')
        proposal_ratios = pipeline_config.get_param('anchor_generator_aspect_ratios')
        proposal_scales = pipeline_config.get_param('anchor_generator_scales')
        anchors_count = len(proposal_ratios) * len(proposal_scales)

        # Convolution/matmul node that produces classes predictions
        # Permute result of the tensor with classes permissions so it will be in a correct layout for Softmax
        predictions_node = match.single_input_node(1)[0].in_node(0).in_node(0)
        permute_predictions_op = Permute(graph, dict(order=np.array([0, 2, 3, 1])))
        permute_predictions_node = permute_predictions_op.create_node([], dict(name=predictions_node.name + '/Permute'))
        insert_node_after(predictions_node, permute_predictions_node, 0)

        # create constant input with the image height, width and scale H and scale W (if present) required for Proposal
        const_op = Const(graph, dict(value=np.array([[input_height, input_width, 1]], dtype=np.float32)))
        const_node = const_op.create_node([], dict(name='proposal_const_image_size'))

        reshape_classes_op = Reshape(graph, dict(dim=np.array([0, -1, 2])))
        reshape_classes_node = reshape_classes_op.create_node([permute_predictions_node],
                                                              dict(name='reshape_FirstStageBoxPredictor_class'))
        update_attrs(reshape_classes_node, 'shape_attrs', 'dim')

        softmax_conf_op = Softmax(graph, dict(axis=1))
        softmax_conf_node = softmax_conf_op.create_node([reshape_classes_node],
                                                        dict(name='FirstStageBoxPredictor_softMax_class'))

        reshape_softmax_op = Reshape(graph, dict(dim=np.array([1, anchors_count, 2, -1])))
        reshape_softmax_node = reshape_softmax_op.create_node([softmax_conf_node], dict(name='reshape_softmax_class'))
        update_attrs(reshape_softmax_node, 'shape_attrs', 'dim')

        permute_reshape_softmax_op = Permute(graph, dict(order=np.array([0, 1, 3, 2])))
        permute_reshape_softmax_node = permute_reshape_softmax_op.create_node([reshape_softmax_node], dict(
            name=reshape_softmax_node.name + '/Permute'))

        # implement custom reshape infer function because we need to know the input convolution node output dimension
        # sizes but we can know it only after partial infer
        reshape_permute_op = Reshape(graph,
                                     dict(dim=np.ones([4]), anchors_count=anchors_count, conv_node=predictions_node))
        reshape_permute_op.attrs['old_infer'] = reshape_permute_op.attrs['infer']
        reshape_permute_op.attrs['infer'] = __class__.classes_probabilities_reshape_shape_infer
        reshape_permute_node = reshape_permute_op.create_node([permute_reshape_softmax_node],
                                                              dict(name='Reshape_Permute_Class'))
        update_attrs(reshape_permute_node, 'shape_attrs', 'dim')

        proposal_op = ProposalOp(graph, dict(min_size=1,
                                             framework='tensorflow',
                                             pre_nms_topn=2 ** 31 - 1,
                                             box_size_scale=5,
                                             box_coordinate_scale=10,
                                             post_nms_topn=max_proposals,
                                             feat_stride=pipeline_config.get_param('features_extractor_stride'),
                                             ratio=proposal_ratios,
                                             scale=proposal_scales,
                                             base_size=pipeline_config.get_param('anchor_generator_base_size'),
                                             nms_thresh=pipeline_config.get_param('first_stage_nms_iou_threshold')))
        proposal_node = proposal_op.create_node([reshape_permute_node,
                                                 match.single_input_node(0)[0].in_node(0).in_node(0),
                                                 const_node],
                                                dict(name=proposal_op.attrs['type']))

        proposal_reshape_4d_op = Reshape(graph, dict(dim=np.array([max_proposals, 1, 1, 5])))
        proposal_reshape_4d_node = proposal_reshape_4d_op.create_node([proposal_node], dict(name="reshape_4d"))
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
        predictions_node = conv_op.create_node([proposal_reshape_4d_node, conv_filter_const_node], dict(name="conv"))
        update_ie_fields(graph.node[predictions_node.id])

        proposal_reshape_2d_op = Reshape(graph, dict(dim=np.array([max_proposals, 5])))
        proposal_reshape_2d_node = proposal_reshape_2d_op.create_node([predictions_node], dict(name="reshape_2d"))
        # set specific name for this Reshape operation so we can use it in the DetectionOutput replacer
        proposal_reshape_2d_node['name'] = 'swapped_proposals'

        # the TF implementation of ROIPooling with bi-linear filtration need proposals scaled by image size
        proposal_scale_const = np.array([1.0, 1 / input_height, 1 / input_width, 1 / input_height, 1 / input_width],
                                        dtype=np.float32)
        proposal_scale_const_op = Const(graph, dict(value=proposal_scale_const))
        proposal_scale_const_node = proposal_scale_const_op.create_node([], dict(name='Proposal_scale_const'))

        scale_proposals_op = Eltwise(graph, dict(operation='mul'))
        scale_proposals_node = scale_proposals_op.create_node([proposal_reshape_2d_node, proposal_scale_const_node],
                                                              dict(name='scale_proposals'))

        feature_extractor_output_nodes = scope_output_nodes(graph, 'FirstStageFeatureExtractor')
        if len(feature_extractor_output_nodes) != 1:
            raise Error("Failed to determine FirstStageFeatureExtractor output node to connect it to the ROIPooling."
                        "Found the following nodes: {}".format([node.name for node in feature_extractor_output_nodes]))

        roi_pooling_op = ROIPooling(graph, dict(method="bilinear", framework="tensorflow", spatial_scale=1,
                                                pooled_h=roi_pool_size, pooled_w=roi_pool_size))
        roi_pooling_node = roi_pooling_op.create_node([feature_extractor_output_nodes[0], scale_proposals_node],
                                                      dict(name='ROI_pooling'))

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


class ObjectDetectionAPISSDPostprocessorReplacement(FrontReplacementFromConfigFileSubGraph):
    replacement_id = 'ObjectDetectionAPISSDPostprocessorReplacement'

    def run_after(self):
        return [ObjectDetectionAPIPreprocessorReplacement]

    def run_before(self):
        # the replacer is uses node of type "RealDiv" as one of the start points, but Model Optimizer replaces nodes of
        # type "RealDiv" with a new ones, so it is necessary to replace the sub-graph before replacing the "RealDiv"
        # nodes
        return [Div, StandaloneConstEraser]

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
        argv = graph.graph['cmd_params']
        if argv.tensorflow_object_detection_api_pipeline_config is None:
            raise Error(missing_param_error)
        pipeline_config = PipelineConfig(argv.tensorflow_object_detection_api_pipeline_config)

        # reshape confidences to 4D before applying sigmoid activation
        expand_dims_op = ExpandDims(graph, {'expand_axis': [1]})
        # do not convert from NHWC to NCHW this node shape
        expand_dims_node = expand_dims_op.add_node(dict(name='do_ExpandDims_conf', nchw_layout=True))

        post_processing_function = pipeline_config.get_param('postprocessing_score_converter')
        if post_processing_function == 'SOFTMAX':
            # softmax to be applied to the confidence
            softmax_conf_op = Softmax(graph, dict(axis=1, nchw_layout=True))
            activation_conf_node = softmax_conf_op.create_node([expand_dims_node],
                                                               dict(name='do_softmax_conf',
                                                                    nchw_layout=True))
        elif post_processing_function == 'SIGMOID':
            # sigmoid activation function to be applied to the confidence
            sigmoid_conf_op = Activation(graph, dict(operation='sigmoid'))
            activation_conf_node = sigmoid_conf_op.create_node([expand_dims_node],
                                                               dict(name='do_sigmoid_conf',
                                                                    nchw_layout=True))
        elif post_processing_function == 'IDENTITY':
            # in case of Identity do nothing and just use result from the input node
            activation_conf_node = expand_dims_node
        else:
            raise Error('Unknown post-processing activation function "{}".'.format(post_processing_function))

        # IE DetectionOutput layer consumes flattened tensors
        # reshape operation to flatten locations tensor
        reshape_loc_op = Reshape(graph, {'dim': np.array([0, -1])})
        reshape_loc_node = reshape_loc_op.add_node(dict(name='do_reshape_loc'))

        # IE DetectionOutput layer consumes flattened tensors
        # reshape operation to flatten confidence tensor
        reshape_conf_op = Reshape(graph, {'dim': np.array([0, -1])})
        reshape_conf_node = reshape_conf_op.add_node(dict(name='do_reshape_conf'))

        # create DetectionOutput Node object from Op class
        detection_output_op = DetectionOutput(graph, match.custom_replacement_desc.custom_attributes)
        detection_output_op.attrs['old_infer'] = detection_output_op.attrs['infer']
        detection_output_op.attrs['infer'] = __class__.do_infer
        detection_output_node = detection_output_op.add_node(
            dict(name=detection_output_op.attrs['type'],
                 clip=1,
                 confidence_threshold=pipeline_config.get_param('postprocessing_score_threshold'),
                 top_k=pipeline_config.get_param('postprocessing_max_detections_per_class'),
                 keep_top_k=pipeline_config.get_param('postprocessing_max_total_detections'),
                 nms_threshold=pipeline_config.get_param('postprocessing_iou_threshold')))

        # create internal edges of the sub-graph. In this case we add edges to connect input port 0 and 1 of the
        # detection output with output of reshape of locations and reshape of confidence
        create_edge(activation_conf_node, reshape_conf_node, 0, 0)
        create_edge(reshape_loc_node, detection_output_node, 0, 0)
        create_edge(reshape_conf_node, detection_output_node, 0, 1)
        return {'detection_output_node': detection_output_node, 'reshape_conf_node': expand_dims_node,
                'reshape_loc_node': reshape_loc_node}

    @staticmethod
    def do_infer(node: Node):
        prior_boxes = node.in_node(2).value
        assert prior_boxes is not None
        # these are default variances values
        variance = np.array([[0.1, 0.1, 0.2, 0.2]])
        # replicating the variance values for all prior-boxes
        variances = np.tile(variance, [prior_boxes.shape[-2], 1])
        # DetectionOutput in the Inference Engine expects the prior-boxes in the following layout: (values, variances)
        prior_boxes = prior_boxes.reshape([-1, 4])
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

        for node_name in node.graph.nodes():
            node = Node(node.graph, node_name)
            if node.has_and_set('swap_xy_count') and len(node.out_nodes()) != node['swap_xy_count']:
                raise Error('The weights were swapped for node "{}", but this weight was used in other nodes.'.format(
                    node.name))


class ObjectDetectionAPIOutputReplacement(FrontReplacementFromConfigFileGeneral):
    """
    This replacer is used to cut-off the network by specified nodes for models generated with Object Detection API.
    The custom attribute for the replacer contains one value for key "outputs". This string is a comma separated list
    of outputs alternatives. Each output alternative is a '|' separated list of node name which could be outputs. The
    first node from each alternative that exits in the graph is chosen. Others are ignored.
    For example, if the "outputs" is equal to the following string:

        "Reshape_16,SecondStageBoxPredictor_1/Conv_3/BiasAdd|SecondStageBoxPredictor_1/Conv_1/BiasAdd"

    then the "Reshape_16" will be an output if it exists in the graph. The second output will be
    SecondStageBoxPredictor_1/Conv_3/BiasAdd if it exist in the graph, if not then
    SecondStageBoxPredictor_1/Conv_1/BiasAdd will be output if it exists in the graph.
    """
    replacement_id = 'ObjectDetectionAPIOutputReplacement'

    def run_before(self):
        return [ObjectDetectionAPIPreprocessorReplacement]

    def transform_graph(self, graph: nx.MultiDiGraph, replacement_descriptions: dict):
        if graph.graph['cmd_params'].output is not None:
            log.warning('User defined output nodes are specified. Skip the graph cut-off by the '
                        'ObjectDetectionAPIOutputReplacement.')
            return
        outputs = []
        outputs_string = replacement_descriptions['outputs']
        for alternatives in outputs_string.split(','):
            for out_node_name in alternatives.split('|'):
                if graph.has_node(out_node_name):
                    outputs.append(out_node_name)
                    break
                else:
                    log.debug('A node "{}" does not exist in the graph. Do not add it as output'.format(out_node_name))
        _outputs = output_user_data_repack(graph, outputs)
        add_output_ops(graph, _outputs, graph.graph['inputs'])
