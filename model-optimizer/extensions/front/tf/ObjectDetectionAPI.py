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

import logging as log
from math import sqrt

import numpy as np

from extensions.back.InsertLayoutPropagationTransposes import mark_as_correct_data_layout, \
    mark_input_as_in_correct_layout, \
    mark_output_as_in_correct_layout
from extensions.front.Pack import Pack
from extensions.front.TransposeOrderNormalizer import TransposeOrderNormalizer
from extensions.front.div import Div
from extensions.front.standalone_const_eraser import StandaloneConstEraser
from extensions.front.sub import Sub
from extensions.front.tf.CropAndResizeReplacement import CropAndResizeReplacement
from extensions.front.tf.Unpack import Unpack
from extensions.ops.DetectionOutput import DetectionOutput
from extensions.ops.ReduceOps import ReduceMean
from extensions.ops.activation_ops import Sigmoid
from extensions.ops.elementwise import Mul
from extensions.ops.parameter import Parameter
from extensions.ops.priorbox_clustered import PriorBoxClusteredOp
from extensions.ops.proposal import ProposalOp
from extensions.ops.psroipooling import PSROIPoolingOp
from extensions.ops.transpose import Transpose
from mo.front.common.layout import get_batch_dim, get_height_dim, get_width_dim
from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.weights import swap_weights_xy
from mo.front.extractor import output_user_data_repack, add_output_ops
from mo.front.subgraph_matcher import SubgraphMatch
from mo.front.tf.graph_utils import add_activation_function_after_node, add_convolution_to_swap_xy_coordinates, \
    squeeze_reshape_and_concat, add_fake_background_loc, create_op_node_with_second_input
from mo.front.tf.replacement import FrontReplacementFromConfigFileSubGraph, FrontReplacementFromConfigFileGeneral
from mo.graph.graph import Graph, Node
from mo.ops.concat import Concat
from mo.ops.const import Const
from mo.ops.crop import Crop
from mo.ops.op import PermuteAttrs
from mo.ops.result import Result
from mo.ops.reshape import Reshape
from mo.ops.roipooling import ROIPooling
from mo.ops.shape import Shape
from mo.ops.softmax import Softmax
from mo.utils.error import Error
from mo.utils.graph import backward_bfs_for_operation, bfs_search
from mo.utils.pipeline_config import PipelineConfig

missing_param_error = 'To convert the model specify path to the pipeline configuration file which was used to ' \
                      'generate the model. Please use "--tensorflow_object_detection_api_pipeline_config" option:\n' \
                      '--tensorflow_object_detection_api_pipeline_config "<path_to_pipeline.config>"\nIf you have ' \
                      'downloaded the model file from the Object Detection Model zoo repository then this file is ' \
                      'located in the archive with frozen model and called "pipeline.config".\nIf you did not use ' \
                      'this command line parameter before that means that you are using currently deprecated ' \
                      'TensorFlow* Object Detection API models conversion mechanism.'


def _value_or_raise(match: SubgraphMatch, pipeline_config: PipelineConfig, key: str):
    """
    Returns value from the 'custom_attributes' of the 'match' object or pipeline_config associated with a key 'key'.
    If the value doesn't exist then raise error.
    :param match: SubgraphMatch object containing 'custom_attributes'.
    :param pipeline_config: PipelineConfig object with parsed values.
    :param key: key to search for.
    :return: the requested value.
    """
    if match and key in match.custom_replacement_desc.custom_attributes:
        return match.custom_replacement_desc.custom_attributes[key]
    value = pipeline_config.get_param(key)
    if value is None:
        raise Error('The sub-graph replacer "[REPLACEMENT_ID]" was not able to find the value for key "{}" in the '
                    'pipeline configuration file specified with the --tensorflow_object_detection_api_pipeline_config '
                    'command line parameter. Update the sub-graph replacement configuration file specified with the '
                    '--tensorflow_use_custom_operations_config command line parameter by adding key "{}" with required '
                    'value to the "custom_attributes" dictionary of the "[REPLACEMENT_ID]" replacer.'.format(key, key))
    return value


def _find_ssd_head_node(graph: Graph, ssd_head_index: int, head_type: str):
    """
    Finds the SSD head node with index 'ssd_head_index' in the topology. The parameter 'head_type' specifies what type
    of the head is requested: with box predictions or class predictions.
    :param graph: graph with the topology.
    :param ssd_head_index: index of the SSD head.
    :param head_type: either 'box' or 'class' string specifying type of the SSD head node.
    :return: the requested Node or None if node is not found.
    """
    if head_type == 'box':
        possible_node_names = ['BoxPredictor_%d/BoxEncodingPredictor/BiasAdd' % ssd_head_index,
                               'WeightSharedConvolutionalBoxPredictor/BoxPredictor/BiasAdd' if ssd_head_index == 0 else
                               'WeightSharedConvolutionalBoxPredictor_%d/BoxPredictor/BiasAdd' % ssd_head_index]
    elif head_type == 'class':
        possible_node_names = ['BoxPredictor_%d/ClassPredictor/BiasAdd' % ssd_head_index,
                               'WeightSharedConvolutionalBoxPredictor/ClassPredictor/BiasAdd' if ssd_head_index == 0
                               else 'WeightSharedConvolutionalBoxPredictor_%d/ClassPredictor/BiasAdd' % ssd_head_index]
    else:
        raise Error('SSD heads can be of type "box" and "class" only.')

    head_node = None
    for head_node_name in possible_node_names:
        if head_node_name in graph.nodes():
            assert (head_node is None)  # only one of the possible node names should exist in the graph
            head_node = Node(graph, head_node_name)
    return head_node


def _variance_from_pipeline_config(pipeline_config: PipelineConfig):
    """
    Generates a numpy array with variances values from the pipeline_config object. The order of the elements is the
    following: variance x, variance y, variance box width, variance box height.
    :param pipeline_config: pipeline_config object to get variances from.
    :return: the numpy array with variances.
    """
    return 1.0 / np.array([pipeline_config.get_param('frcnn_variance_x'),
                           pipeline_config.get_param('frcnn_variance_y'),
                           pipeline_config.get_param('frcnn_variance_width'),
                           pipeline_config.get_param('frcnn_variance_height')])


def _skip_node_of_type(node: Node, node_ops_to_skip: list):
    """
    Skips nodes of specified ops starting from node 'node'.
    :param node: node to start skipping Identity nodes.
    :return: node of the op
    """
    # skip the Identity node
    while len(node.out_edges()) == 1 and node.op in node_ops_to_skip:
        node = node.out_node()
    return node


def _relax_reshape_nodes(graph: Graph, pipeline_config: PipelineConfig):
    """
    Finds the 'Reshape' operations following the SSD head nodes which have hard-coded output dimensions and replaces
    them with new ones with one of the dimensions sizes equal to -1. This function is used to make TF OD API SSD models
    reshape-able.
    :param graph: graph with the topology.
    :param pipeline_config: PipelineConfig object with parsed values.
    :return: None
    """
    num_classes = pipeline_config.get_param('num_classes')
    num_layers = pipeline_config.get_param('ssd_anchor_generator_num_layers')
    if num_layers is None:
        num_layers = pipeline_config.get_param('multiscale_anchor_generator_max_level') - \
                     pipeline_config.get_param('multiscale_anchor_generator_min_level') + 1
    for ssd_head_ind in range(num_layers):
        input_node = _find_ssd_head_node(graph, ssd_head_ind, 'box')
        assert (input_node is not None)
        old_reshape_node = _skip_node_of_type(input_node.out_node(), ['Identity', 'FakeQuantWithMinMaxVars'])
        assert old_reshape_node.op == 'Reshape'
        reshape_size_node = Const(graph, {'value': int64_array([0, -1, 1, 4])}).create_node([])
        new_reshape_op = Reshape(graph, {'name': input_node.id + '/Reshape'})
        new_reshape_node = new_reshape_op.create_node([input_node, reshape_size_node])
        old_reshape_node.replace_node(new_reshape_node)

        # fix hard-coded value for the number of items in tensor produced by the convolution to make topology reshapable
        input_node = _find_ssd_head_node(graph, ssd_head_ind, 'class')
        assert (input_node is not None)
        old_reshape_node = _skip_node_of_type(input_node.out_node(), ['Identity', 'FakeQuantWithMinMaxVars'])
        assert old_reshape_node.op == 'Reshape'
        reshape_size_node_2 = Const(graph, {'value': int64_array([0, -1, num_classes + 1])}).create_node([])
        new_reshape_op_2 = Reshape(graph, {'name': input_node.id + '/Reshape'})
        new_reshape_node_2 = new_reshape_op_2.create_node([input_node, reshape_size_node_2])
        old_reshape_node.replace_node(new_reshape_node_2)


def _create_prior_boxes_node(graph: Graph, pipeline_config: PipelineConfig):
    """
    The function creates one or several PriorBoxClustered nodes based on information from the pipeline configuration
    files. The PriorBoxClustered nodes get input data from SSD 'heads' and from the placeholder node (just to get
    input image size).
    :param graph: graph with the topology.
    :param pipeline_config: PipelineConfig object with parsed values.
    :return: node generating prior boxes.
    """
    min_scale = pipeline_config.get_param('ssd_anchor_generator_min_scale')
    max_scale = pipeline_config.get_param('ssd_anchor_generator_max_scale')
    num_layers = pipeline_config.get_param('ssd_anchor_generator_num_layers')
    aspect_ratios = pipeline_config.get_param('ssd_anchor_generator_aspect_ratios')
    if not isinstance(aspect_ratios, list):
        aspect_ratios = [aspect_ratios]

    # prior boxes have to be generated using the image size used for training
    image_height = pipeline_config.get_param('resizer_image_height')
    image_width = pipeline_config.get_param('resizer_image_width')
    min_im_shape = min(image_height, image_width)
    _base_anchor_height = pipeline_config.get_param('ssd_anchor_generator_base_anchor_height')
    _base_anchor_width = pipeline_config.get_param('ssd_anchor_generator_base_anchor_width')
    base_anchor_size = [min_im_shape / image_height * _base_anchor_height,
                        min_im_shape / image_width * _base_anchor_width]
    reduce_boxes_in_lowest_layer = True
    if pipeline_config.get_param('ssd_anchor_generator_reduce_lowest') is not None:
        reduce_boxes_in_lowest_layer = pipeline_config.get_param('ssd_anchor_generator_reduce_lowest')

    if pipeline_config.get_param('ssd_anchor_generator_scales') is not None:
        scales = pipeline_config.get_param('ssd_anchor_generator_scales') + [1.0]
    else:
        scales = [min_scale + (max_scale - min_scale) * i / (num_layers - 1) for i in range(num_layers)] + [1.0]

    prior_box_nodes = []
    for ssd_head_ind in range(num_layers):
        ssd_head_node = _find_ssd_head_node(graph, ssd_head_ind, 'box')
        assert (ssd_head_node is not None)

        if ssd_head_ind == 0 and reduce_boxes_in_lowest_layer:
            widths = [0.1, min_scale * sqrt(2.0), min_scale * sqrt(0.5)]
            heights = [0.1, min_scale / sqrt(2.0), min_scale / sqrt(0.5)]
        else:
            widths = [scales[ssd_head_ind] * sqrt(ar) for ar in aspect_ratios]
            heights = [scales[ssd_head_ind] / sqrt(ar) for ar in aspect_ratios]

            interpolated_scale_ar = pipeline_config.get_param('ssd_anchor_generator_interpolated_scale_aspect_ratio')
            if interpolated_scale_ar > 0.0:
                widths += [sqrt(scales[ssd_head_ind] * scales[ssd_head_ind + 1]) * interpolated_scale_ar]
                heights += [sqrt(scales[ssd_head_ind] * scales[ssd_head_ind + 1]) / interpolated_scale_ar]
        widths = [w * image_width * base_anchor_size[1] for w in widths]
        heights = [h * image_height * base_anchor_size[0] for h in heights]

        variance = _variance_from_pipeline_config(pipeline_config)
        prior_box_op = PriorBoxClusteredOp(graph, {'width': np.array(widths), 'height': np.array(heights),
                                                   'clip': 0, 'flip': 0, 'variance': variance, 'offset': 0.5,
                                                   })
        # connect the PriorBoxClustered node with the "Cast" node of the Placeholder node because the pass that removes
        # Cast operations is executed in the middle phase and it will fail when there are several consumers of the
        # Placeholder
        prior_box_node = prior_box_op.create_node([ssd_head_node, Node(graph, 'image_tensor').out_node(0)],
                                                  {'name': 'PriorBoxClustered_{}'.format(ssd_head_ind)})
        prior_box_nodes.append(prior_box_node)
    if len(prior_box_nodes) == 1:
        return prior_box_nodes[0]
    else:
        concat_prior_boxes_op = Concat(graph, {'axis': -1, 'in_ports_count': len(prior_box_nodes)})
        return concat_prior_boxes_op.create_node(prior_box_nodes, {'name': 'ConcatPriorBoxesClustered'})


def _create_multiscale_prior_boxes_node(graph: Graph, pipeline_config: PipelineConfig):
    """
    The function creates one or several PriorBoxClustered nodes based on information from the pipeline configuration
    files. The PriorBoxClustered nodes get input data from SSD 'heads' and from the placeholder node (just to get
    input image size).
    :param graph: graph with the topology.
    :param pipeline_config: PipelineConfig object with parsed values.
    :return: node generating prior boxes.
    """
    min_level = pipeline_config.get_param('multiscale_anchor_generator_min_level')
    max_level = pipeline_config.get_param('multiscale_anchor_generator_max_level')
    anchor_scale = pipeline_config.get_param('multiscale_anchor_generator_anchor_scale')
    aspect_ratios = pipeline_config.get_param('multiscale_anchor_generator_aspect_ratios')
    scales_per_octave = pipeline_config.get_param('multiscale_anchor_generator_scales_per_octave')

    prior_box_nodes = []
    scales = [2 ** (float(scale) / scales_per_octave) for scale in range(scales_per_octave)]
    for level in range(min_level, max_level + 1):
        base_anchor_size = 2 ** level * anchor_scale

        ssd_head_ind = level - min_level
        ssd_head_node = _find_ssd_head_node(graph, ssd_head_ind, 'box')
        assert (ssd_head_node is not None)

        widths = [base_anchor_size * scale * sqrt(ar) for ar in aspect_ratios for scale in scales]
        heights = [base_anchor_size * scale / sqrt(ar) for ar in aspect_ratios for scale in scales]

        variance = _variance_from_pipeline_config(pipeline_config)
        prior_box_op = PriorBoxClusteredOp(graph, {'width': np.array(widths), 'height': np.array(heights),
                                                   'clip': 0, 'flip': 0, 'variance': variance,
                                                   'offset': 0.5,
                                                   })
        # connect the PriorBoxClustered node with the "Cast" node of the Placeholder node because the pass that removes
        # Cast operations is executed in the middle phase and it will fail when there are several consumers of the
        # Placeholder
        prior_box_node = prior_box_op.create_node([ssd_head_node, Node(graph, 'image_tensor').out_node(0)],
                                                  {'name': 'PriorBoxClustered_{}'.format(ssd_head_ind)})
        prior_box_nodes.append(prior_box_node)
    if len(prior_box_nodes) == 1:
        return prior_box_nodes[0]
    else:
        concat_prior_boxes_op = Concat(graph, {'axis': -1, 'in_ports_count': len(prior_box_nodes)})
        return concat_prior_boxes_op.create_node(prior_box_nodes, {'name': 'ConcatPriorBoxesClustered'})


def skip_nodes_by_condition(current_node: Node, condition: callable):
    while condition(current_node):
        current_node = current_node.in_node()
    return current_node


def calculate_shape_keeping_aspect_ratio(height: int, width: int, min_size: int, max_size: int):
    """
    The function scales spatial sizes of the image keeping aspect ratio to satisfy provided requirements.
    The behavior of this function is equivalent to the output shape calculation of the Preprocessor block of TensorFlow
    Object Detection API models with keep aspect ratio resizer.
    :param height: input height.
    :param width: input width.
    :param min_size: size limit.
    :param max_size: size limit.
    :return: the tuple with scaled image height, width.
    """
    ratio_min = min_size / min(height, width)
    ratio_max = max_size / max(height, width)
    ratio = min(ratio_min, ratio_max)
    return int(round(height * ratio)), int(round(width * ratio))


def calculate_placeholder_spatial_shape(graph: Graph, match: SubgraphMatch, pipeline_config: PipelineConfig):
    """
    The function calculates the preprocessed shape of the input image for a TensorFlow Object Detection API model.
    It uses various sources to calculate it:
    1. The shape passed using the '--input_shape' command line parameter.
    2. The values from the pipeline configuration file describing Preprocessor block of the topology:
        a. If the fixed size resizer is used then size passed via '--input_shape' can override them, but Model Optimizer
           prints warning. If the '--input_shape' is not defined then use values from the pipeline configuration file.
        b. If the keep aspect ratio resizer is used then scale the size passed via '--input_shape' using the provided
           limits. If the '--input_shape' is not defined then use shape as (min_dimension_size, min_dimension_size)
           defined in the pipeline configuration file.
    :param graph: graph with the topology.
    :param match: the object containing matching sub-graph and custom attributes from the sub-graph replacement file.
    :param pipeline_config: the object contain information from the pipeline configuration file.
    :return: tuple (height, width) of the placeholder shape.
    """
    height = None
    width = None
    user_shapes = graph.graph['user_shapes']

    if 'preprocessed_image_height' in match.custom_replacement_desc.custom_attributes or 'preprocessed_image_width' in \
            match.custom_replacement_desc.custom_attributes:
        log.error('The "preprocessed_image_height" or "preprocessed_image_width" is specified in the sub-graph '
                  'replacement configuration file but they are ignored. Please, specify desired input shape using the '
                  '"--input_shape" command line parameter.', extra={'is_warning': True})

    user_defined_height = None
    user_defined_width = None
    if user_shapes and 'image_tensor' in user_shapes and user_shapes['image_tensor']:
        user_defined_shape = user_shapes['image_tensor'][0]['shape']
        if user_defined_shape is not None:
            user_defined_height = user_defined_shape[1]
            user_defined_width = user_defined_shape[2]

    resizer_height = pipeline_config.get_param('resizer_image_height')
    resizer_width = pipeline_config.get_param('resizer_image_width')
    if resizer_height and resizer_width:
        log.debug('The model resizes image to a fixed shape: ({}, {})'.format(resizer_height, resizer_width))

    resizer_min_dimension = pipeline_config.get_param('resizer_min_dimension')
    resizer_max_dimension = pipeline_config.get_param('resizer_max_dimension')
    if resizer_min_dimension and resizer_max_dimension:
        log.debug('The model resizes image using keep aspect ratio with minimum size {}, maximum size {}.'.format(
            resizer_min_dimension, resizer_max_dimension))

    # if the model is created with an input image resizer to a fixed shape
    if resizer_width and resizer_height:
        if user_defined_height and user_defined_width:
            if user_defined_width != resizer_width or user_defined_width != resizer_width:
                log.error('The model expects that the input image is resized to a fixed shape ({}, {}), but the shape '
                          'provided with the "--input_shape" command line parameter is different ({}, {}).'.format(
                    resizer_height, resizer_width, user_defined_height, user_defined_width), extra={'is_warning': True})
            height = user_defined_height
            width = user_defined_width
        else:
            height = resizer_height
            width = resizer_width

    # if the model is created with an input image resizer keeping aspect ratio
    if resizer_min_dimension and resizer_max_dimension:
        print('[ WARNING ] Model Optimizer removes pre-processing block of the model which resizes image keeping '
              'aspect ratio. The Inference Engine does not support dynamic image size so the Intermediate '
              'Representation file is generated with the input image size of a fixed size.')
        if user_defined_height and user_defined_width:
            scaled_height, scaled_width = calculate_shape_keeping_aspect_ratio(user_defined_height,
                                                                               user_defined_width,
                                                                               resizer_min_dimension,
                                                                               resizer_max_dimension)
            if scaled_height != user_defined_height or scaled_width != scaled_width:
                log.error('The model resizes the input image keeping aspect ratio with min dimension {}, max '
                          'dimension {}. The provided input height {}, width {} is transformed to height {}, width '
                          '{}.'.format(resizer_min_dimension, resizer_max_dimension, user_defined_height,
                                       user_defined_width, scaled_height, scaled_width), extra={'is_warning': True})
            height = scaled_height
            width = scaled_width
        else:
            height = width = resizer_min_dimension
            print('Specify the "--input_shape" command line parameter to override the default shape which is equal to '
                  '({}, {}).'.format(height, width))

    if height is None or width is None:
        raise Error('Failed to determine the placeholder shape.')
    return height, width


class ObjectDetectionAPIPreprocessorReplacement(FrontReplacementFromConfigFileSubGraph):
    """
    The class replaces the "Preprocessor" block resizing input image and applying mean/scale values. Only nodes related
    to applying mean/scaling values are kept.
    """
    replacement_id = 'ObjectDetectionAPIPreprocessorReplacement'

    def run_before(self):
        return [Pack, Sub, TransposeOrderNormalizer]

    def nodes_to_remove(self, graph: Graph, match: SubgraphMatch):
        new_nodes_to_remove = match.matched_nodes_names()
        # do not remove nodes that perform input image scaling and mean value subtraction
        for node_to_keep in ('Preprocessor/sub', 'Preprocessor/sub/y', 'Preprocessor/mul', 'Preprocessor/mul/x'):
            if node_to_keep in new_nodes_to_remove:
                new_nodes_to_remove.remove(node_to_keep)
        return new_nodes_to_remove

    def generate_sub_graph(self, graph: Graph, match: SubgraphMatch):
        argv = graph.graph['cmd_params']
        layout = graph.graph['layout']
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

        # set default value of the batch size to 1 if user didn't specify batch size and input shape
        batch_dim = get_batch_dim(layout, 4)
        if argv.batch is None and placeholder_node.shape[batch_dim] == -1:
            placeholder_node.shape[batch_dim] = 1
        height, width = calculate_placeholder_spatial_shape(graph, match, pipeline_config)
        placeholder_node.shape[get_height_dim(layout, 4)] = height
        placeholder_node.shape[get_width_dim(layout, 4)] = width

        # save the pre-processed image spatial sizes to be used in the other replacers
        graph.graph['preprocessed_image_height'] = placeholder_node.shape[get_height_dim(layout, 4)]
        graph.graph['preprocessed_image_width'] = placeholder_node.shape[get_width_dim(layout, 4)]

        to_float_node = placeholder_node.out_node(0)
        if not to_float_node.has('op') or to_float_node.op != 'Cast':
            raise Error('The output of the node "{}" is not Cast operation. Cannot apply replacer.'.format(
                initial_input_node_name))

        # connect to_float_node directly with node performing scale on mean value subtraction
        if mul_node is None:
            graph.create_edge(to_float_node, sub_node, 0, 0)
        else:
            graph.create_edge(to_float_node, mul_node, 0, 1)

        print('The Preprocessor block has been removed. Only nodes performing mean value subtraction and scaling (if'
              ' applicable) are kept.')
        return {}


class ObjectDetectionAPIDetectionOutputReplacement(FrontReplacementFromConfigFileSubGraph):
    """
    Replaces the sub-graph that is equal to the DetectionOutput layer from Inference Engine. This replacer is used for
    Faster R-CNN, R-FCN and Mask R-CNN topologies conversion.
    The replacer uses a value of the custom attribute 'coordinates_swap_method' from the sub-graph replacement
    configuration file to choose how to swap box coordinates of the 0-th input of the generated DetectionOutput layer.
    Refer to the code for more details.
    """
    replacement_id = 'ObjectDetectionAPIDetectionOutputReplacement'

    def run_before(self):
        return [ObjectDetectionAPIMaskRCNNROIPoolingSecondReplacement, Unpack, Sub, TransposeOrderNormalizer]

    def run_after(self):
        return [ObjectDetectionAPIProposalReplacement, CropAndResizeReplacement]

    def nodes_to_remove(self, graph: Graph, match: SubgraphMatch):
        new_nodes_to_remove = match.matched_nodes_names().copy()
        outputs = ['detection_boxes', 'detection_scores', 'num_detections']
        for output in outputs:
            children = Node(graph, output).out_nodes()
            if len(children) != 1:
                log.warning('Output {} has {} children. It should have only one output: with op==`Result`'
                            ''.format(output, len(children)))
            elif children[list(children.keys())[0]].op == 'Result':
                new_nodes_to_remove.append(children[list(children.keys())[0]].id)
            else:
                continue
        new_nodes_to_remove.extend(outputs)
        return new_nodes_to_remove

    def output_edges_match(self, graph: Graph, match: SubgraphMatch, new_sub_graph: dict):
        # the DetectionOutput in IE produces single tensor, but in TF it produces four tensors, so we need to create
        # only one output edge match
        return {match.output_node(0)[0].id: new_sub_graph['detection_output_node'].id}

    def generate_sub_graph(self, graph: Graph, match: SubgraphMatch):
        argv = graph.graph['cmd_params']
        if argv.tensorflow_object_detection_api_pipeline_config is None:
            raise Error(missing_param_error)
        pipeline_config = PipelineConfig(argv.tensorflow_object_detection_api_pipeline_config)

        num_classes = _value_or_raise(match, pipeline_config, 'num_classes')
        max_proposals = _value_or_raise(match, pipeline_config, 'first_stage_max_proposals')
        activation_function = _value_or_raise(match, pipeline_config, 'postprocessing_score_converter')

        activation_conf_node = add_activation_function_after_node(graph, match.single_input_node(1)[0].in_node(0),
                                                                  activation_function)

        # IE DetectionOutput layer consumes flattened tensors so need add a Reshape layer.
        # The batch value of the input tensor is not equal to the batch of the topology, so it is not possible to use
        # "0" value in the Reshape layer attribute to refer to the batch size, but we know how to
        # calculate the second dimension so the batch value will be deduced from it with help of "-1".
        reshape_conf_node = create_op_node_with_second_input(graph, Reshape,
                                                             int64_array([-1, (num_classes + 1) * max_proposals]),
                                                             dict(name='do_reshape_conf'), activation_conf_node)

        mark_as_correct_data_layout(reshape_conf_node)

        # Workaround for TransposeForReshape pass.
        # We looking for first not Reshape-typed node before match.single_input_node(0)[0].in_node(0).
        # And add  reshape_loc node after this first not Reshape-typed node.
        current_node = skip_nodes_by_condition(match.single_input_node(0)[0].in_node(0),
                                               lambda x: x['kind'] == 'op' and x.has_and_set('reinterp_shape'))

        reshape_loc_node = create_op_node_with_second_input(graph, Reshape, int64_array([-1, num_classes, 1, 4]),
                                                            dict(name='reshape_loc'), current_node)
        mark_as_correct_data_layout(reshape_loc_node)

        # constant node with variances
        variances_const_op = Const(graph, dict(value=_variance_from_pipeline_config(pipeline_config)))
        variances_const_node = variances_const_op.create_node([])

        # TF produces locations tensor without boxes for background.
        # Inference Engine DetectionOutput layer requires background boxes so we generate them
        loc_node = add_fake_background_loc(graph, reshape_loc_node)
        PermuteAttrs.set_permutation(reshape_loc_node, loc_node, None)

        # reshape locations tensor to 2D so it could be passed to Eltwise which will be converted to ScaleShift
        reshape_loc_2d_node = create_op_node_with_second_input(graph, Reshape, int64_array([-1, 4]),
                                                               dict(name='reshape_locs_2d'), loc_node)
        mark_as_correct_data_layout(reshape_loc_2d_node)

        # element-wise multiply locations with variances
        eltwise_locs_op = Mul(graph, dict())
        eltwise_locs_node = eltwise_locs_op.create_node([reshape_loc_2d_node, variances_const_node],
                                                        dict(name='scale_locs'))

        # IE DetectionOutput layer consumes flattened tensors so need add a Reshape layer.
        # The batch value of the input tensor is not equal to the batch of the topology, so it is not possible to use
        # "0" value in the Reshape layer attribute to refer to the batch size, but we know how to
        # calculate the second dimension so the batch value will be deduced from it with help of "-1".
        reshape_loc_do_op = Reshape(graph, dict(name='do_reshape_locs'))

        custom_attributes = match.custom_replacement_desc.custom_attributes
        coordinates_swap_method = 'add_convolution'
        if 'coordinates_swap_method' not in custom_attributes:
            log.error('The ObjectDetectionAPIDetectionOutputReplacement sub-graph replacement configuration file '
                      'must contain "coordinates_swap_method" in the "custom_attributes" dictionary. Two values are '
                      'supported: "swap_weights" and "add_convolution". The first one should be used when there is '
                      'a MatMul or Conv2D node before the "SecondStagePostprocessor" block in the topology. With this '
                      'solution the weights of the MatMul or Conv2D nodes are permutted, simulating the swap of XY '
                      'coordinates in the tensor. The second could be used in any other cases but it is worse in terms '
                      'of performance because it adds the Conv2D node which performs permutting of data. Since the '
                      'attribute is not defined the second approach is used by default.')
        else:
            coordinates_swap_method = custom_attributes['coordinates_swap_method']
        supported_swap_methods = ['swap_weights', 'add_convolution']
        if coordinates_swap_method not in supported_swap_methods:
            raise Error('Unsupported "coordinates_swap_method" defined in the sub-graph replacement configuration '
                        'file. Supported methods are: {}'.format(', '.join(supported_swap_methods)))

        if coordinates_swap_method == 'add_convolution':
            swapped_locs_node = add_convolution_to_swap_xy_coordinates(graph, eltwise_locs_node, 4)
            reshape_loc_do_node = reshape_loc_do_op.create_node([swapped_locs_node])
        else:
            reshape_loc_do_node = reshape_loc_do_op.create_node([eltwise_locs_node])

        reshape_loc_do_dims = Const(graph, {'value': int64_array([-1, (num_classes + 1) * max_proposals * 4]),
                                            'name': reshape_loc_do_node.name + '/Dim'}).create_node()
        reshape_loc_do_dims.out_port(0).connect(reshape_loc_do_node.in_port(1))

        mark_as_correct_data_layout(reshape_loc_do_node)

        # find Proposal output which has the data layout as in TF: YXYX coordinates without batch indices.
        proposal_nodes_ids = [node_id for node_id, attrs in graph.nodes(data=True)
                              if 'name' in attrs and attrs['name'] == 'crop_proposals']
        if len(proposal_nodes_ids) != 1:
            raise Error("Found the following nodes '{}' with name 'crop_proposals' but there should be exactly 1. "
                        "Looks like ObjectDetectionAPIProposalReplacement replacement didn't work.".
                        format(proposal_nodes_ids))
        proposal_node = Node(graph, proposal_nodes_ids[0])

        # check whether it is necessary to permute proposals coordinates before passing them to the DetectionOutput
        # currently this parameter is set for the RFCN topologies
        if 'swap_proposals' in custom_attributes and custom_attributes['swap_proposals']:
            proposal_node = add_convolution_to_swap_xy_coordinates(graph, proposal_node, 4)

        # reshape priors boxes as Detection Output expects
        reshape_priors_node = create_op_node_with_second_input(graph, Reshape, int64_array([-1, 1, max_proposals * 4]),
                                                               dict(name='DetectionOutput_reshape_priors_'),
                                                               proposal_node)
        mark_as_correct_data_layout(reshape_priors_node)

        detection_output_op = DetectionOutput(graph, {})
        if coordinates_swap_method == 'swap_weights':
            # update infer function to re-pack weights
            detection_output_op.attrs['old_infer'] = detection_output_op.attrs['infer']
            detection_output_op.attrs['infer'] = __class__.do_infer
        for key in ('clip_before_nms', 'clip_after_nms'):
            if key in match.custom_replacement_desc.custom_attributes:
                detection_output_op.attrs[key] = int(match.custom_replacement_desc.custom_attributes[key])

        detection_output_node = detection_output_op.create_node(
            [reshape_loc_do_node, reshape_conf_node, reshape_priors_node],
            dict(name=detection_output_op.attrs['type'], share_location=0, variance_encoded_in_target=1,
                 code_type='caffe.PriorBoxParameter.CENTER_SIZE', pad_mode='caffe.ResizeParameter.CONSTANT',
                 resize_mode='caffe.ResizeParameter.WARP',
                 num_classes=num_classes,
                 confidence_threshold=_value_or_raise(match, pipeline_config, 'postprocessing_score_threshold'),
                 top_k=_value_or_raise(match, pipeline_config, 'postprocessing_max_detections_per_class'),
                 keep_top_k=_value_or_raise(match, pipeline_config, 'postprocessing_max_total_detections'),
                 nms_threshold=_value_or_raise(match, pipeline_config, 'postprocessing_iou_threshold')))
        # sets specific name to the node so we can find it in other replacers
        detection_output_node.name = 'detection_output'

        output_op = Result(graph, dict(name='do_OutputOp'))
        output_op.create_node([detection_output_node])

        print('The graph output nodes "num_detections", "detection_boxes", "detection_classes", "detection_scores" '
              'have been replaced with a single layer of type "Detection Output". Refer to IR catalogue in the '
              'documentation for information about this layer.')

        return {'detection_output_node': detection_output_node}

    @staticmethod
    def do_infer(node):
        node.old_infer(node)
        # compared to the IE's DetectionOutput, the TF keeps the locations in YXYX, need to get back to the XYXY
        # for last matmul/Conv2D that operate the locations need to swap the X and Y for output feature weights & biases
        swap_weights_xy(backward_bfs_for_operation(node.in_node(0), ['MatMul', 'Conv2D']))


class ObjectDetectionAPIMaskRCNNROIPoolingSecondReplacement(FrontReplacementFromConfigFileSubGraph):
    replacement_id = 'ObjectDetectionAPIMaskRCNNROIPoolingSecondReplacement'

    def run_after(self):
        return [ObjectDetectionAPIProposalReplacement]

    def output_edges_match(self, graph: Graph, match: SubgraphMatch, new_sub_graph: dict):
        return {match.output_node(0)[0].id: new_sub_graph['roi_pooling_node'].id}

    def generate_sub_graph(self, graph: Graph, match: SubgraphMatch):
        argv = graph.graph['cmd_params']
        if argv.tensorflow_object_detection_api_pipeline_config is None:
            raise Error(missing_param_error)
        pipeline_config = PipelineConfig(argv.tensorflow_object_detection_api_pipeline_config)
        roi_pool_size = _value_or_raise(match, pipeline_config, 'initial_crop_size')

        detection_output_nodes_ids = [node_id for node_id, attrs in graph.nodes(data=True)
                                      if 'name' in attrs and attrs['name'] == 'detection_output']
        if len(detection_output_nodes_ids) != 1:
            raise Error("Found the following nodes '{}' with 'detection_output' but there should be exactly 1.".
                        format(detection_output_nodes_ids))
        detection_output_node = Node(graph, detection_output_nodes_ids[0])

        # add reshape of Detection Output so it can be an output of the topology
        reshape_detection_output_2d_node = create_op_node_with_second_input(graph, Reshape, int64_array([-1, 7]),
                                                                            dict(name='reshape_do_2d'),
                                                                            detection_output_node)
        mark_as_correct_data_layout(reshape_detection_output_2d_node)

        # adds special node of type "Output" that is a marker for the output nodes of the topology
        output_op = Result(graph, dict(name='do_reshaped_OutputOp'))
        output_node = output_op.create_node([reshape_detection_output_2d_node])

        # add attribute 'output_sort_order' so it will be used as a key to sort output nodes before generation of IR
        output_node.in_edge()['data_attrs'].append('output_sort_order')
        output_node.in_edge()['output_sort_order'] = [('detection_boxes', 0)]

        # creates two Crop operations which get input from the DetectionOutput layer, cuts of slices of data with class
        # ids and probabilities and produce a tensor with batch ids and bounding boxes only (as it is expected by the
        # ROIPooling layer)
        crop_batch_op = Crop(graph, dict(axis=int64_array([3]), offset=int64_array([0]), dim=int64_array([1]),
                                         nchw_layout=True))
        crop_batch_node = crop_batch_op.create_node([detection_output_node], dict(name='crop_do_batch_ids'))

        crop_coordinates_op = Crop(graph, dict(axis=int64_array([3]), offset=int64_array([3]), dim=int64_array([4]),
                                               nchw_layout=True))
        crop_coordinates_node = crop_coordinates_op.create_node([detection_output_node], dict(name='crop_do_coords'))

        concat_op = Concat(graph, dict(axis=3))
        concat_node = concat_op.create_node([crop_batch_node, crop_coordinates_node], dict(name='batch_and_coords',
                                                                                           nchw_layout=True))

        # reshape bounding boxes as required by ROIPooling
        reshape_do_node = create_op_node_with_second_input(graph, Reshape, int64_array([-1, 5]),
                                                           dict(name='reshape_do'), concat_node)
        mark_as_correct_data_layout(reshape_do_node)

        roi_pooling_op = ROIPooling(graph, dict(method="bilinear", spatial_scale=1, pooled_h=roi_pool_size,
                                                pooled_w=roi_pool_size))
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

    def transform_graph(self, graph: Graph, replacement_descriptions):
        output_node = None
        op_outputs = [n for n, d in graph.nodes(data=True) if 'op' in d and d['op'] == 'Result']
        for op_output in op_outputs:
            last_node = Node(graph, op_output).in_node(0)
            if last_node.name.startswith('SecondStageBoxPredictor'):
                sigmoid_node = Sigmoid(graph, dict(name=last_node.id + '/sigmoid')).create_node([last_node])
                sigmoid_node.name = 'masks'

                if output_node is not None:
                    raise Error('Identified two possible outputs from the topology. Cannot proceed.')
                # add special node of type "Output" that is a marker for the output nodes of the topology
                output_op = Result(graph, dict(name=sigmoid_node.name + '/OutputOp'))
                output_node = output_op.create_node([sigmoid_node])

        print('The predicted masks are produced by the "masks" layer for each bounding box generated with a '
              '"detection_output" layer.\n Refer to IR catalogue in the documentation for information '
              'about the DetectionOutput layer and Inference Engine documentation about output data interpretation.\n'
              'The topology can be inferred using dedicated demo "mask_rcnn_demo".')


class ObjectDetectionAPIProposalReplacement(FrontReplacementFromConfigFileSubGraph):
    """
    This class replaces sub-graph of operations with Proposal layer and additional layers transforming
    tensors from layout of TensorFlow to layout required by Inference Engine.
    Refer to comments inside the function for more information about performed actions.
    """
    replacement_id = 'ObjectDetectionAPIProposalReplacement'

    def run_after(self):
        return [ObjectDetectionAPIPreprocessorReplacement]

    def run_before(self):
        return [Sub, CropAndResizeReplacement, TransposeOrderNormalizer]

    def output_edges_match(self, graph: Graph, match: SubgraphMatch, new_sub_graph: dict):
        return {match.output_node(0)[0].id: new_sub_graph['proposal_node'].id}

    def nodes_to_remove(self, graph: Graph, match: SubgraphMatch):
        new_list = match.matched_nodes_names().copy()
        # do not remove nodes that produce box predictions and class predictions
        new_list.remove(match.single_input_node(0)[0].id)
        new_list.remove(match.single_input_node(1)[0].id)
        return new_list

    def generate_sub_graph(self, graph: Graph, match: SubgraphMatch):
        argv = graph.graph['cmd_params']
        if argv.tensorflow_object_detection_api_pipeline_config is None:
            raise Error(missing_param_error)
        pipeline_config = PipelineConfig(argv.tensorflow_object_detection_api_pipeline_config)

        max_proposals = _value_or_raise(match, pipeline_config, 'first_stage_max_proposals')
        proposal_ratios = _value_or_raise(match, pipeline_config, 'anchor_generator_aspect_ratios')
        proposal_scales = _value_or_raise(match, pipeline_config, 'anchor_generator_scales')
        anchors_count = len(proposal_ratios) * len(proposal_scales)

        # Convolution/matmul node that produces classes predictions
        # Transpose result of the tensor with classes permissions so it will be in a correct layout for Softmax
        predictions_node = backward_bfs_for_operation(match.single_input_node(1)[0], ['Add'])[0]

        reshape_classes_node = create_op_node_with_second_input(graph, Reshape, int64_array([0, anchors_count, 2, -1]),
                                                                dict(name='predictions/Reshape'))
        predictions_node.insert_node_after(reshape_classes_node, 0)
        mark_as_correct_data_layout(reshape_classes_node)

        softmax_conf_op = Softmax(graph, dict(axis=2, nchw_layout=True, name=reshape_classes_node.id + '/Softmax'))
        softmax_conf_node = softmax_conf_op.create_node([reshape_classes_node])

        order_const = Const(graph, dict(value=int64_array([0, 2, 1, 3]),
                                        name=softmax_conf_node.name + '/TransposeOrder')).create_node()
        permute_reshape_softmax_op = Transpose(graph, dict())
        permute_reshape_softmax_node = permute_reshape_softmax_op.create_node([softmax_conf_node, order_const], dict(
            name=softmax_conf_node.name + '/Transpose'))
        mark_input_as_in_correct_layout(permute_reshape_softmax_node, 1)
        mark_output_as_in_correct_layout(permute_reshape_softmax_node, 0)

        initial_shape_op = Shape(graph, dict(name=predictions_node.id + '/Shape'))
        initial_shape_node = initial_shape_op.create_node([predictions_node])

        reshape_permute_op = Reshape(graph, dict(name='Reshape_Transpose_Class'))
        reshape_permute_node = reshape_permute_op.create_node([permute_reshape_softmax_node, initial_shape_node])
        mark_input_as_in_correct_layout(reshape_permute_node, 0)
        mark_output_as_in_correct_layout(reshape_permute_node, 0)

        variance_height = pipeline_config.get_param('frcnn_variance_height')
        variance_width = pipeline_config.get_param('frcnn_variance_width')
        variance_x = pipeline_config.get_param('frcnn_variance_x')
        variance_y = pipeline_config.get_param('frcnn_variance_y')
        anchor_generator_height_stride = pipeline_config.get_param('anchor_generator_height_stride')
        anchor_generator_width_stride = pipeline_config.get_param('anchor_generator_width_stride')
        anchor_generator_height = pipeline_config.get_param('anchor_generator_height')
        anchor_generator_width = pipeline_config.get_param('anchor_generator_width')

        if variance_height != variance_width:
            log.error('The values for variance for height "{}" is not equal to variance for width "{}". The detection '
                      'results will be inaccurate.'.format(variance_height, variance_width))
        if variance_x != variance_y:
            log.error('The values for variance for x "{}" is not equal to variance for y "{}". The detection '
                      'results will be inaccurate.'.format(variance_x, variance_y))
        if anchor_generator_height_stride != anchor_generator_width_stride:
            log.error('The values for the anchor generator height stride "{}" is not equal to the anchor generator '
                      'width stride "{}". The detection results will be inaccurate.'.format(
                anchor_generator_height_stride, anchor_generator_width_stride))
        if anchor_generator_height != anchor_generator_width:
            log.error('The values for the anchor generator height "{}" is not equal to the anchor generator width '
                      'stride "{}". The detection results will be inaccurate.'.format(anchor_generator_height,
                                                                                      anchor_generator_width))

        proposal_op = ProposalOp(graph, dict(min_size=1,
                                             framework='tensorflow',
                                             pre_nms_topn=2 ** 31 - 1,
                                             box_size_scale=variance_height,
                                             box_coordinate_scale=variance_x,
                                             post_nms_topn=max_proposals,
                                             feat_stride=anchor_generator_height_stride,
                                             ratio=proposal_ratios,
                                             scale=proposal_scales,
                                             normalize=1,
                                             base_size=anchor_generator_height,
                                             nms_thresh=_value_or_raise(match, pipeline_config,
                                                                        'first_stage_nms_iou_threshold')))
        for key in ('clip_before_nms', 'clip_after_nms'):
            if key in match.custom_replacement_desc.custom_attributes:
                proposal_op.attrs[key] = int(match.custom_replacement_desc.custom_attributes[key])

        anchors_node = backward_bfs_for_operation(match.single_input_node(0)[0], ['Add'])[0]

        # creates input to store input image height, width and scales (usually 1.0s)
        # the batch size for this input is fixed because it is allowed to pass images of the same size only as input
        input_op_with_image_size = Parameter(graph, dict(shape=int64_array([1, 3]), fixed_batch=True))
        input_with_image_size_node = input_op_with_image_size.create_node([], dict(name='image_info'))

        proposal_node = proposal_op.create_node([reshape_permute_node, anchors_node, input_with_image_size_node],
                                                dict(name='proposals'))

        if 'do_not_swap_proposals' in match.custom_replacement_desc.custom_attributes and \
                match.custom_replacement_desc.custom_attributes['do_not_swap_proposals']:
            swapped_proposals_node = proposal_node
        else:
            swapped_proposals_node = add_convolution_to_swap_xy_coordinates(graph, proposal_node, 5)

        proposal_reshape_2d_node = create_op_node_with_second_input(graph, Reshape, int64_array([-1, 5]),
                                                                    dict(name="reshape_swap_proposals_2d"),
                                                                    swapped_proposals_node)
        mark_input_as_in_correct_layout(proposal_reshape_2d_node, 0)

        # feed the CropAndResize node with a correct boxes information produced with the Proposal layer
        # find the first CropAndResize node in the BFS order
        crop_and_resize_nodes_ids = [node_id for node_id in bfs_search(graph, [match.single_input_node(0)[0].id]) if
                                     graph.node[node_id]['op'] == 'CropAndResize']
        assert len(crop_and_resize_nodes_ids) != 0, "Didn't find any CropAndResize nodes in the graph."
        if 'do_not_swap_proposals' not in match.custom_replacement_desc.custom_attributes or not \
                match.custom_replacement_desc.custom_attributes['do_not_swap_proposals']:
            crop_and_resize_node = Node(graph, crop_and_resize_nodes_ids[0])
            # set a marker that the input with box coordinates has been pre-processed so the CropAndResizeReplacement
            # transform doesn't try to merge the second and the third inputs
            crop_and_resize_node['inputs_preprocessed'] = True
            graph.remove_edge(crop_and_resize_node.in_node(1).id, crop_and_resize_node.id)
            graph.create_edge(proposal_reshape_2d_node, crop_and_resize_node, out_port=0, in_port=1)

        tf_proposal_reshape_4d_node = create_op_node_with_second_input(graph, Reshape,
                                                                       int64_array([-1, 1, max_proposals, 5]),
                                                                       dict(name="reshape_proposal_4d"),
                                                                       swapped_proposals_node)

        crop_op = Crop(graph, dict(axis=int64_array([3]), offset=int64_array([1]), dim=int64_array([4]),
                                   nchw_layout=True))
        crop_node = crop_op.create_node([tf_proposal_reshape_4d_node], dict(name='crop_proposals'))

        mark_as_correct_data_layout(tf_proposal_reshape_4d_node)

        tf_proposals_crop_reshape_3d_node = create_op_node_with_second_input(graph, Reshape, int64_array([0, -1, 4]),
                                                                             dict(name="reshape_crop_3d"), crop_node)
        mark_input_as_in_correct_layout(tf_proposals_crop_reshape_3d_node, 0)
        return {'proposal_node': tf_proposals_crop_reshape_3d_node}


class ObjectDetectionAPISSDPostprocessorReplacement(FrontReplacementFromConfigFileSubGraph):
    replacement_id = 'ObjectDetectionAPISSDPostprocessorReplacement'

    def run_after(self):
        return [ObjectDetectionAPIPreprocessorReplacement]

    def run_before(self):
        # the replacer uses node of type "RealDiv" as one of the start points, but Model Optimizer replaces nodes of
        # type "RealDiv" with a new ones, so it is necessary to replace the sub-graph before replacing the "RealDiv"
        # nodes
        return [Div, StandaloneConstEraser, TransposeOrderNormalizer]

    def output_edges_match(self, graph: Graph, match: SubgraphMatch, new_sub_graph: dict):
        # the DetectionOutput in IE produces single tensor, but in TF it produces two tensors, so create only one output
        # edge match
        return {match.output_node(0)[0].id: new_sub_graph['detection_output_node'].id}

    def generate_sub_graph(self, graph: Graph, match: SubgraphMatch):
        argv = graph.graph['cmd_params']
        if argv.tensorflow_object_detection_api_pipeline_config is None:
            raise Error(missing_param_error)
        pipeline_config = PipelineConfig(argv.tensorflow_object_detection_api_pipeline_config)
        num_classes = _value_or_raise(match, pipeline_config, 'num_classes')

        # reshapes confidences to 4D before applying activation function and do not convert from NHWC to NCHW this node
        expand_dims_node = create_op_node_with_second_input(graph, Reshape, int64_array([0, 1, -1, num_classes + 1]),
                                                            {'name': 'do_ExpandDims_conf'})
        expand_dims_node.in_port(0).connect(match.input_nodes(1)[0][0].in_node(0).out_port(0))

        mark_as_correct_data_layout(expand_dims_node)

        activation_function = _value_or_raise(match, pipeline_config, 'postprocessing_score_converter')
        activation_conf_node = add_activation_function_after_node(graph, expand_dims_node, activation_function)

        # IE DetectionOutput layer consumes flattened tensors
        # reshape operation to flatten locations tensor
        reshape_loc_node = create_op_node_with_second_input(graph, Reshape, int64_array([0, -1]),
                                                            {'name': 'do_reshape_loc'})

        current_node = skip_nodes_by_condition(match.input_nodes(0)[0][0].in_node(0),
                                               lambda x: x.op == 'Identity' or x.has_and_set('reinterp_shape'))
        reshape_loc_node.in_port(0).connect(current_node.out_port(0))
        mark_as_correct_data_layout(reshape_loc_node)

        # IE DetectionOutput layer consumes flattened tensors
        # reshape operation to flatten confidence tensor
        reshape_conf_node = create_op_node_with_second_input(graph, Reshape, int64_array([0, -1]),
                                                             {'name': 'do_reshape_conf'}, activation_conf_node)
        mark_as_correct_data_layout(reshape_conf_node)

        custom_attributes = match.custom_replacement_desc.custom_attributes
        if ('disable_prior_boxes_layers_generator' not in custom_attributes or
            not custom_attributes['disable_prior_boxes_layers_generator']) and \
            (pipeline_config.get_param('ssd_anchor_generator_num_layers') is not None or
                pipeline_config.get_param('multiscale_anchor_generator_min_level') is not None):
            # change the Reshape operations with hardcoded number of output elements of the convolution nodes to be
            # reshapable
            _relax_reshape_nodes(graph, pipeline_config)

            # create PriorBoxClustered nodes instead of a constant value with prior boxes so the model could be reshaped
            if pipeline_config.get_param('ssd_anchor_generator_num_layers') is not None:
                priors_node = _create_prior_boxes_node(graph, pipeline_config)
            elif pipeline_config.get_param('multiscale_anchor_generator_min_level') is not None:
                priors_node = _create_multiscale_prior_boxes_node(graph, pipeline_config)
        else:
            log.info('The anchor generator is not known. Save constant with prior-boxes to IR.')
            priors_node = match.input_nodes(2)[0][0].in_node(0)

        # creates DetectionOutput Node object from Op class
        detection_output_op = DetectionOutput(graph, match.custom_replacement_desc.custom_attributes)
        detection_output_op.attrs['old_infer'] = detection_output_op.attrs['infer']
        detection_output_op.attrs['infer'] = __class__.do_infer
        detection_output_node = detection_output_op.create_node(
            [reshape_loc_node, reshape_conf_node, priors_node],
            dict(name=detection_output_op.attrs['type'],
                 clip=1,
                 confidence_threshold=_value_or_raise(match, pipeline_config, 'postprocessing_score_threshold'),
                 top_k=_value_or_raise(match, pipeline_config, 'postprocessing_max_detections_per_class'),
                 keep_top_k=_value_or_raise(match, pipeline_config, 'postprocessing_max_total_detections'),
                 nms_threshold=_value_or_raise(match, pipeline_config, 'postprocessing_iou_threshold')))

        return {'detection_output_node': detection_output_node}

    @staticmethod
    def do_infer(node: Node):
        prior_boxes = node.in_node(2).value
        if prior_boxes is not None:
            argv = node.graph.graph['cmd_params']
            if argv.tensorflow_object_detection_api_pipeline_config is None:
                raise Error(missing_param_error)
            pipeline_config = PipelineConfig(argv.tensorflow_object_detection_api_pipeline_config)
            variance = _variance_from_pipeline_config(pipeline_config)
            # replicating the variance values for all prior-boxes
            variances = np.tile(variance, [prior_boxes.shape[-2], 1])
            # DetectionOutput Inference Engine expects the prior-boxes in the following layout: (values, variances)
            prior_boxes = prior_boxes.reshape([-1, 4])
            prior_boxes = np.concatenate((prior_boxes, variances), 0)
            # compared to the IE's DetectionOutput, the TF keeps the prior-boxes in YXYX, need to get back to the XYXY
            prior_boxes = np.concatenate((prior_boxes[:, 1:2], prior_boxes[:, 0:1],
                                          prior_boxes[:, 3:4], prior_boxes[:, 2:3]), 1)
            #  adding another dimensions, as the prior-boxes are expected as 3d tensors
            prior_boxes = prior_boxes.reshape((1, 2, -1))
            node.in_node(2).shape = int64_array(prior_boxes.shape)
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
        return [ObjectDetectionAPIPreprocessorReplacement, TransposeOrderNormalizer]

    def transform_graph(self, graph: Graph, replacement_descriptions: dict):
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


class ObjectDetectionAPIPSROIPoolingReplacement(FrontReplacementFromConfigFileSubGraph):
    replacement_id = 'ObjectDetectionAPIPSROIPoolingReplacement'

    def run_after(self):
        return [ObjectDetectionAPIProposalReplacement, TransposeOrderNormalizer]

    def output_edges_match(self, graph: Graph, match: SubgraphMatch, new_sub_graph: dict):
        return {match.output_node(0)[0].id: new_sub_graph['output_node'].id}

    def generate_sub_graph(self, graph: Graph, match: SubgraphMatch):
        argv = graph.graph['cmd_params']
        if argv.tensorflow_object_detection_api_pipeline_config is None:
            raise Error(missing_param_error)
        pipeline_config = PipelineConfig(argv.tensorflow_object_detection_api_pipeline_config)
        num_classes = _value_or_raise(match, pipeline_config, 'num_classes')

        input_node = match.input_nodes(0)[0][0].in_node(0)
        if 'class_predictions' in input_node.id:
            psroipooling_output_dim = num_classes + 1
        else:
            psroipooling_output_dim = num_classes * 4

        num_spatial_bins_height = pipeline_config.get_param('num_spatial_bins_height')
        num_spatial_bins_width = pipeline_config.get_param('num_spatial_bins_width')
        crop_height = pipeline_config.get_param('crop_height')
        crop_width = pipeline_config.get_param('crop_width')
        if crop_height != crop_width:
            raise Error('Different "crop_height" and "crop_width" parameters from the pipeline config are not '
                        'supported: {} vs {}'.format(crop_height, crop_width))

        if 'reshape_swap_proposals_2d' in graph.nodes():
            reshape_swap_proposals_node = Node(graph, 'reshape_swap_proposals_2d')
        else:
            swap_proposals_node = add_convolution_to_swap_xy_coordinates(graph, Node(graph, 'proposals'), 5)
            reshape_swap_proposals_node = create_op_node_with_second_input(graph, Reshape, int64_array([-1, 5]),
                                                                           {'name': 'reshape_swap_proposals_2d'},
                                                                           swap_proposals_node)
            mark_input_as_in_correct_layout(reshape_swap_proposals_node, 0)

        psroipooling_node = PSROIPoolingOp(graph, {'name': input_node.soft_get('name') + '/PSROIPooling',
                                                   'output_dim': psroipooling_output_dim,
                                                   'group_size': crop_width / num_spatial_bins_width,
                                                   'spatial_bins_x': num_spatial_bins_width,
                                                   'spatial_bins_y': num_spatial_bins_height,
                                                   'mode': 'bilinear',
                                                   'spatial_scale': 1,
                                                   }).create_node([input_node, reshape_swap_proposals_node])

        reduce_node = create_op_node_with_second_input(graph, ReduceMean, int64_array([1, 2]),
                                                       {'name': 'mean', 'keep_dims': True}, psroipooling_node)

        output_node = match.output_node(0)[0].out_node()
        if len(output_node.in_ports()) == 2 and not output_node.in_port(1).disconnected():
            output_node.in_port(1).disconnect()  # disconnect the second input to make "erase_node" function work
        graph.erase_node(match.output_node(0)[0].out_node())

        return {'output_node': reduce_node}


class ObjectDetectionAPIConstValueOverride(FrontReplacementFromConfigFileGeneral):
    """
    Transforms allows to override specific constant values in the topology. The replacement description configuration
    file contains list of tuples describing the desired replacements specified in the "replacements" key of the
    "custom_attributes". The first element in the tuple is the initial node name of the graph with constant value. The
    second element is the name of the parameter from the pipeline configuration file which stores new value.

    Usage example. The Faster-RCNNs topologies has constant node with the number specifying maximum generated proposals.
    This value is specified in the pipeline configuration file in the parameter 'first_stage_max_proposals' and is
    saved as a constant node in the generated topology. If the parameter is modified from it's original value then the
    topology will be incorrect because the number 'first_stage_max_proposals' is used in the transforms of this file is
    no more equal to the 'first_stage_max_proposals' saved as a constant.
    """
    replacement_id = 'ObjectDetectionAPIConstValueOverride'

    def run_before(self):
        return [ObjectDetectionAPIPreprocessorReplacement, TransposeOrderNormalizer]

    def transform_graph(self, graph: Graph, replacement_descriptions: dict):
        argv = graph.graph['cmd_params']
        if argv.tensorflow_object_detection_api_pipeline_config is None:
            raise Error(missing_param_error)
        pipeline_config = PipelineConfig(argv.tensorflow_object_detection_api_pipeline_config)
        for (node_id, pipeline_config_name) in replacement_descriptions['replacements']:
            if node_id not in graph.nodes():
                log.debug('Node with id {} does not exist in the graph'.format(node_id))
                continue
            node = Node(graph, node_id)
            if not node.has_valid('value'):
                log.debug('Node with id {} does not have value'.format(node_id))
                continue
            node.value = np.array(pipeline_config.get_param(pipeline_config_name))
            node.value = node.value.reshape(node.shape)
