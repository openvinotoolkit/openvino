# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
The file contains necessary transformations to convert models created with a TensorFlow Object Detection framework from
the https://github.com/tensorflow/models/blob/master/research/object_detection/ repository. There is a dedicated
OpenVINO document describing overall procedure of conversion these models with the Model Optimizer:
https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_Object_Detection_API_Models.html

Conversion of most of the TF OD API models requires execution of several transformations defined in this file. The list
of transformations to be executed for a particular model type (meta-architecture) is defined in the transformation
configuration JSON file located in the "openvino/tools/mo/front/tf/" directory. A file should be specified using the
"--transformations_config" command line parameter. An additional parameter
"--tensorflow_object_detection_api_pipeline_config" should be specified with the path to the pipeline.config used for
the model training.

Refer to the code comments of a particular transformation for the explanation of its purpose and low-level
implementation details.
"""
import collections
import logging as log
from math import sqrt

import numpy as np

from openvino.tools.mo.front.Pack import Pack
from openvino.tools.mo.front.TransposeOrderNormalizer import TransposeOrderNormalizer
from openvino.tools.mo.front.split_normalizer import SqueezeAxis
from openvino.tools.mo.front.tf.CropAndResizeReplacement import CropAndResizeReplacement
from openvino.tools.mo.front.FakeQuantWithMinMaxVars import FakeQuantWithMinMaxVarsToQuantize
from openvino.tools.mo.front.tf.MapFNTransformation import MapFNInputSlicing, MapFNOutputConcatenation,\
    TensorListOutputConcatenation
from openvino.tools.mo.front.tf.TFSliceToSlice import TFSliceToSliceReplacer
from openvino.tools.mo.front.tf.pad_tf_to_pad import PadTFToPad
from openvino.tools.mo.middle.InsertLayoutPropagationTransposes import mark_as_correct_data_layout, \
    mark_input_as_in_correct_layout, mark_output_as_in_correct_layout
from openvino.tools.mo.ops.Cast import Cast
from openvino.tools.mo.ops.DetectionOutput import DetectionOutput
from openvino.tools.mo.ops.ReduceOps import ReduceMean
from openvino.tools.mo.ops.activation_ops import Sigmoid
from openvino.tools.mo.ops.elementwise import Mul, Sub, Add, Div
from openvino.tools.mo.ops.gather import Gather
from openvino.tools.mo.ops.parameter import Parameter
from openvino.tools.mo.ops.priorbox_clustered import PriorBoxClusteredOp
from openvino.tools.mo.ops.proposal import ProposalOp
from openvino.tools.mo.ops.psroipooling import PSROIPoolingOp
from openvino.tools.mo.ops.split import Split
from openvino.tools.mo.ops.transpose import Transpose
from openvino.tools.mo.front.common.layout import get_batch_dim, get_height_dim, get_width_dim
from openvino.tools.mo.front.common.partial_infer.utils import int64_array, dynamic_dimension, mo_array, dynamic_dimension_value
from openvino.tools.mo.front.common.replacement import FrontReplacementPattern
from openvino.tools.mo.front.extractor import output_user_data_repack, add_output_ops
from openvino.tools.mo.front.subgraph_matcher import SubgraphMatch
from openvino.tools.mo.front.tf.custom_subgraph_call import skip_nodes_by_condition
from openvino.tools.mo.front.tf.graph_utils import add_activation_function_after_node, add_convolution_to_swap_xy_coordinates, \
    add_fake_background_loc, create_op_node_with_second_input, create_op_with_const_inputs
from openvino.tools.mo.front.tf.replacement import FrontReplacementFromConfigFileSubGraph, FrontReplacementFromConfigFileGeneral
from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.ops.clamp import AttributedClamp
from openvino.tools.mo.ops.concat import Concat
from openvino.tools.mo.ops.const import Const
from openvino.tools.mo.ops.crop import Crop
from openvino.tools.mo.ops.op import PermuteAttrs
from openvino.tools.mo.ops.reshape import Reshape
from openvino.tools.mo.ops.result import Result
from openvino.tools.mo.ops.roipooling import ROIPooling
from openvino.tools.mo.ops.shape import Shape
from openvino.tools.mo.ops.softmax import Softmax
from openvino.tools.mo.ops.squeeze import Squeeze
from openvino.tools.mo.ops.tile import Tile
from openvino.tools.mo.utils.error import Error
from openvino.tools.mo.utils.graph import backward_bfs_for_operation, bfs_search, clear_tensor_names_info, sub_graph_between_nodes
from openvino.tools.mo.utils.pipeline_config import PipelineConfig
from openvino.tools.mo.utils.shape import node_to_get_shape_value_of_indices

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
                    '--transformations_config command line parameter by adding key "{}" with required '
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
    return 1.0 / mo_array([pipeline_config.get_param('frcnn_variance_x'),
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
        old_reshape_node = _skip_node_of_type(input_node.out_node(),
                                              ['Identity', 'FakeQuantWithMinMaxVars', 'FakeQuantize'])
        assert old_reshape_node.op == 'Reshape'
        reshape_size_node = Const(graph, {'value': int64_array([0, -1, 1, 4])}).create_node([])
        new_reshape_op = Reshape(graph, {'name': input_node.id + '/Reshape'})
        new_reshape_node = new_reshape_op.create_node([input_node, reshape_size_node])
        old_reshape_node.replace_node(new_reshape_node)

        # fix hard-coded value for the number of items in tensor produced by the convolution to make topology reshapable
        input_node = _find_ssd_head_node(graph, ssd_head_ind, 'class')
        assert (input_node is not None)
        old_reshape_node = _skip_node_of_type(input_node.out_node(),
                                              ['Identity', 'FakeQuantWithMinMaxVars', 'FakeQuantize'])

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
        prior_box_op = PriorBoxClusteredOp(graph, {'width': mo_array(widths), 'height': mo_array(heights),
                                                   'clip': 0, 'flip': 0, 'variance': variance, 'offset': 0.5,
                                                   })
        # connect the PriorBoxClustered node with the "Cast" node of the Placeholder node because the pass that removes
        # Cast operations is executed in the middle phase and it will fail when there are several consumers of the
        # Placeholder
        input_node_name = 'image_tensor' if 'image_tensor' in graph.nodes else 'input_tensor'
        prior_box_node = prior_box_op.create_node([ssd_head_node, Node(graph, input_node_name).out_node(0)],
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
        prior_box_op = PriorBoxClusteredOp(graph, {'width': mo_array(widths), 'height': mo_array(heights),
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


def insert_weights_swap_xy_sub_graph(graph: Graph, connection):
    """
    Inserts a sub-graph of operations which does the following:
    1. Reshapes the input tensor (should be convolution weights/biases) to [-1, 2].
    2. Swaps slices of data [:, 0] and [:, 1].
    3. Reshapes tensor to the initial shape.
    """
    weights_producer = connection.get_source()
    name = weights_producer.node.soft_get('name', weights_producer.node.id)

    # this Shape operation must be inferred and constant folded
    origin_shape = Shape(graph, {'name': name + '/OriginShape', 'force_dead_node': True}).create_node()
    origin_shape.in_port(0).connect(weights_producer)

    reshaped = create_op_node_with_second_input(graph, Reshape, int64_array([-1, 2]), {'name': name + '/Reshape2D'})
    reshaped.in_port(0).connect(weights_producer)

    swapped_weight = Gather(graph, {'name': name + '/SwappedWeights'}).create_node()
    gather_indices = Const(graph,
                           {'name': swapped_weight.name + '/Indices', 'value': int64_array([1, 0])}).create_node()
    gather_axis = Const(graph, {'name': swapped_weight.name + '/Axis', 'value': int64_array(1)}).create_node()
    swapped_weight.in_port(0).connect(reshaped.out_port(0))
    swapped_weight.in_port(1).connect(gather_indices.out_port(0))
    swapped_weight.in_port(2).connect(gather_axis.out_port(0))

    reshape_back = Reshape(graph, {'name': name + '/ReshapeBack'}).create_node()
    reshape_back.in_port(0).connect(swapped_weight.out_port(0))
    reshape_back.in_port(1).connect(origin_shape.out_port(0))

    connection.set_source(reshape_back.out_port(0))


def swap_weights_xy(graph: Graph, nodes: list):
    """
    The function changes weights of the nodes from the 'nodes' list which are used with calculations with coordinates of
    some objects. The function should be used when it is necessary to virtually change the layout of data from XY to YX.
    The node from the 'nodes' list should be some sort of convolution node or matrix multiplication.
    The function also swaps weights in the following Add and BiasAdd operations.
    :param graph: graph with the topology.
    :param nodes: list of Node objects to change the weights in them.
    :return: None
    """
    producers_ports = set()
    for node in nodes:
        # need to skip the FakeQuantize node if it exists
        weights_producer = node.in_port(1).get_source()
        if weights_producer.node.soft_get('type') == 'FakeQuantize':
            weights_producer = weights_producer.node.in_port(0).get_source()
        producers_ports.add(weights_producer)

    for producers_port in producers_ports:
        log.debug('Swapping weights for node "{}"'.format(producers_port.node.name))
        insert_weights_swap_xy_sub_graph(graph, producers_port.get_connection())

    for node in nodes:
        # swap biases
        for m in [n.node for n in node.out_port(0).get_destinations()]:
            if m.soft_get('type') in ['Add', 'BiasAdd']:
                insert_weights_swap_xy_sub_graph(graph, m.in_port(1).get_connection())


def calculate_shape_keeping_aspect_ratio(height: int, width: int, min_size: int, max_size: int):
    """
    The function changes spatial sizes of the image keeping aspect ratio to satisfy provided requirements.
    The behavior of this function is equivalent to the output shape calculation of the pre-processor block of TensorFlow
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
           defined in the pipeline configuration file. If the "pad_to_max_dimension" attribute is set to true then the
           output shape will always be (max_dimension_size, max_dimension_size).

    :param graph: graph with the topology.
    :param match: the object containing matching sub-graph and custom attributes from the sub-graph replacement file.
    :param pipeline_config: the object contain information from the pipeline configuration file.
    :return: tuple (height, width) of the placeholder shape.
    """
    height = None
    width = None
    user_shapes = graph.graph['user_shapes']

    if match and ('preprocessed_image_height' in match.custom_replacement_desc.custom_attributes or
                  'preprocessed_image_width' in match.custom_replacement_desc.custom_attributes):
        log.error('The "preprocessed_image_height" or "preprocessed_image_width" is specified in the sub-graph '
                  'replacement configuration file but they are ignored. Please, specify desired input shape using the '
                  '"--input_shape" command line parameter.', extra={'is_warning': True})

    user_defined_height = None
    user_defined_width = None
    input_name = 'input_tensor' if 'input_tensor' in graph.nodes else 'image_tensor'
    if user_shapes and input_name in user_shapes and user_shapes[input_name]:
        user_defined_shape = user_shapes[input_name][0]['shape']
        if user_defined_shape is not None:
            user_defined_height = user_defined_shape[1].get_min_length() if user_defined_shape[1].is_static else dynamic_dimension_value
            user_defined_width = user_defined_shape[2].get_min_length() if user_defined_shape[2].is_static else dynamic_dimension_value

    # the parameters below are set if the fixed_shape_resizer is used
    resizer_height = pipeline_config.get_param('resizer_image_height')
    resizer_width = pipeline_config.get_param('resizer_image_width')
    if resizer_height and resizer_width:
        log.debug('The model resizes image to a fixed shape: ({}, {})'.format(resizer_height, resizer_width))
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

    # the parameters below are set if keep_aspect_ratio_resizer is used
    resizer_min_dimension = pipeline_config.get_param('resizer_min_dimension')
    resizer_max_dimension = pipeline_config.get_param('resizer_max_dimension')
    pad_to_max_dimension = pipeline_config.get_param('pad_to_max_dimension')
    if resizer_min_dimension and resizer_max_dimension:
        log.debug('The model resizes image using keep aspect ratio with minimum size {}, maximum size {}, pad {}.'
                  ''.format(resizer_min_dimension, resizer_max_dimension, pad_to_max_dimension))
        if pad_to_max_dimension:
            if user_defined_height and user_defined_width:
                log.error('The model contains pre-processing block which resizes image keeping aspect ratio with a '
                          'padding to max dimension. The only valid model input image spatial shape after '
                          'pre-processing is ({}, {}). Ignoring the user provided input shapes.'
                          ''.format(resizer_max_dimension, resizer_max_dimension), extra={'is_warning': True})
            height = width = resizer_max_dimension
        else:
            log.error('Model Optimizer removes pre-processing block of the model which resizes image keeping aspect '
                      'ratio. OpenVINO does not support dynamic image size so the Intermediate Representation '
                      'file is generated with the input image size of a fixed size.', extra={'is_warning': True})
            if user_defined_height and user_defined_width:
                scaled_height, scaled_width = calculate_shape_keeping_aspect_ratio(user_defined_height,
                                                                                   user_defined_width,
                                                                                   resizer_min_dimension,
                                                                                   resizer_max_dimension)
                if scaled_height != user_defined_height or scaled_width != user_defined_width:
                    log.error('The model resizes the input image keeping aspect ratio with min dimension {}, max '
                              'dimension {}. The provided input height {}, width {} is transformed to height {}, width '
                              '{}.'.format(resizer_min_dimension, resizer_max_dimension, user_defined_height,
                                           user_defined_width, scaled_height, scaled_width), extra={'is_warning': True})
                height = scaled_height
                width = scaled_width
            else:
                height = width = resizer_min_dimension
                log.error('Specify the "--input_shape" command line parameter to override the default shape which is '
                          'equal to ({}, {}).'.format(height, width), extra={'is_warning': True})

    if height is None or width is None:
        raise Error('Failed to determine the placeholder shape. Unsupported image resizer from the pipeline.config was '
                    'used to create the model.')
    return height, width


def update_parameter_shape(graph: Graph, match: [SubgraphMatch, None]):
    """
    Updates the shape of the model Parameter node based on the user provided input shape or values provided in the
    pipeline.config configuration file used for model training.
    :param graph: model graph
    :param match: Match object with information about matched sub-graph
    :return: tuple with input node names and Parameter Node
    """
    argv = graph.graph['cmd_params']
    if argv.tensorflow_object_detection_api_pipeline_config is None:
        raise Error(missing_param_error)

    pipeline_config = PipelineConfig(argv.tensorflow_object_detection_api_pipeline_config)
    if argv.tensorflow_object_detection_api_pipeline_config is None:
        raise Error(missing_param_error)

    initial_input_node_name = 'input_tensor' if 'input_tensor' in graph.nodes else 'image_tensor'
    if initial_input_node_name not in graph.nodes():
        raise Error('Input node "{}" of the graph is not found. Do not run the Model Optimizer with '
                    '"--input" command line parameter.'.format(initial_input_node_name))
    parameter_node = Node(graph, initial_input_node_name)

    # set default value of the batch size to 1 if user didn't specify batch size and input shape
    layout = graph.graph['layout']
    batch_dim = get_batch_dim(layout, 4)
    if argv.batch is None and parameter_node.shape[batch_dim] is dynamic_dimension:
        parameter_node.shape[batch_dim] = 1
    height, width = calculate_placeholder_spatial_shape(graph, match, pipeline_config)
    parameter_node.shape[get_height_dim(layout, 4)] = height
    parameter_node.shape[get_width_dim(layout, 4)] = width
    return initial_input_node_name, parameter_node


def mark_squeeze_reshape_concat_before_detection_output(start_nodes: list):
    """
    The function looks for Reshape, Concat and Squeeze ops after the 'start_nodes' with 4D output and marks them with
    proper attributes to infer them in original NHWC layout. This is a case of the TensorFlow Object Detection API
    models for the SSD heads output which produces 4D tensor with bounding box deltas.

    :param start_nodes: list of nodes to start search from.
    :return: None
    """
    q = collections.deque()
    visited = set()
    q.extend(start_nodes)
    while len(q) != 0:
        cur_node = q.popleft()
        visited.add(cur_node.id)
        if cur_node.has_valid('type'):
            if cur_node.soft_get('type') == 'DetectionOutput':  # do not go beyond the DetectionOutput node
                continue
            # the input to Reshape comes from Convolution so it will be converted from NCHW to NHWC layout in the
            # InsertLayoutPropagationTransposes transformation. But the output should be kept in the original layout
            if cur_node.soft_get('type') == 'Reshape':
                mark_output_as_in_correct_layout(cur_node, 0)

            # Concat should be inferred in the original layout so the input with concatenation axis should not be
            # updated from NHWC to NCHW layout
            if cur_node.soft_get('type') == 'Concat':
                cur_node.in_port(1).__setattr__('input_permutation', None)
                cur_node['nchw_layout'] = True
                cur_node.out_node(0)['nchw_layout'] = True

            # Squeeze should be inferred in the original layout so the input with squeeze axis should not be updated
            # from NHWC to NCHW layout. The input is marked as in correct layout to prevent from inserting Transpose
            # from NHWC to NCHW.
            if cur_node.soft_get('type') == 'Squeeze':
                cur_node.in_port(1).__setattr__('input_permutation', None)
                mark_input_as_in_correct_layout(cur_node, 0)

        if cur_node.has_port('out', 0):
            [q.append(port.node) for port in cur_node.out_port(0).get_destinations() if port.node.id not in visited]


class ObjectDetectionAPITransformationsStart(FrontReplacementPattern):
    """
    This is a anchor transformation which is used to distinguish TF OD API models related transformations.
    All transformations have a dependency to be executed after this transformation (or some other TF OD API
    transformation which is executed after this one).
    Some transformation which swap convolution weights using the "swap_weights_xy" function relies on the fact that the
    "FakeQuantWithMinMaxVars" operations are decomposed into "FakeQuantize"s.
    """
    enabled = True

    def run_after(self):
        return [FakeQuantWithMinMaxVarsToQuantize]

    def find_and_replace_pattern(self, graph: Graph):
        pass


class ObjectDetectionAPITransformationsFinish(FrontReplacementPattern):
    """
    This is a anchor transformation which is used to separate TF OD API models related transformations.
    All transformations have a dependency to be executed before this transformation (or some other TF OD API
    transformation which is executed before this one).
    1. This anchor transformation is executed before any other standard MO transformations which may break the model
    conversion. For example, PadTFToPad replaces PadTF operation nodes with the Pad operation nodes and re-uses an
    input node defining the pad value. The scope pattern matcher will remove the node defining the pad value and the
    newly created Pad operation become invalid.
    2. Another common issue that some transformations should be executed after TF OD API transformations is that these
    transformations replace some nodes with new nodes but different "id" attribute. Since the pattern matcher is based
    on node "id" (not "name") attribute the matching will be broken.
    3. Some TF OD API transformations mark TF CropAndResize nodes with specific flag which is then handled in the
    CropAndResizeReplacement transformation that is why latter one should be executed after this transformation.
    """
    enabled = True
    # cleanup the graph after applying of TF OD API transformations to remove a lot of unconnected nodes to avoid issues
    # with shape inference
    force_clean_up = True

    def run_before(self):
        return [Pack, TransposeOrderNormalizer, PadTFToPad, SqueezeAxis, TFSliceToSliceReplacer, MapFNInputSlicing,
                MapFNOutputConcatenation, TensorListOutputConcatenation, CropAndResizeReplacement]

    def find_and_replace_pattern(self, graph: Graph):
        pass


def get_specific_ops_with_const_inputs(first_node: Node, allowed_ops: list, forward: bool = True):
    """
    Returns the list with information about consecutive nodes of operation from "allowed_ops".

    :param first_node: The first node (not included) to start looking for nodes from the "allowed_ops" list
    :param allowed_ops: list of allowed operations
    :param forward: flag specifying direction of search
    :return: list of triplets (Node, const_port_index, const_value)
    """
    node = first_node.out_port(0).get_destination().node if forward else first_node.in_port(0).get_source().node
    result = []  # (Node, port # with constant input, value)
    while node.soft_get('op') in allowed_ops:
        num_in_ports = len(node.in_ports())
        assert num_in_ports == 2, 'The node "{}" should have exactly 2 inputs, but it has only {}.' \
                                  ''.format(node.soft_get('name', node.id), num_in_ports)
        for port in (0, 1):
            if node.in_port(port).get_source().node.has_valid('value'):  # this is a constant input to the node
                result.append((node, port, node.in_port(port).get_source().node.value.copy()))
                node = node.out_port(0).get_destination().node if forward else node.in_port(1 - port).get_source().node
                break
    return result


def get_preprocessing_ops(graph: Graph, start_node_id_suffix: str, end_node_id_suffix: str):
    """
    Finds a sequence of pre-processing nodes (Sub, Mul, Div and Add) after the node with the id suffix
    'end_node_id_suffix' or ending with the node with id suffix 'end_node_id_suffix'.

    :param graph: graph to look for pre-processing ops
    :param start_node_id_suffix: suffix of the start node name
    :param end_node_id_suffix: suffix of the end node name
    :return: the list with pre-processing nodes information and flag specifying nodes position
    """
    start_node = None
    end_node = None
    for node in graph.get_op_nodes():
        if node.id.endswith(start_node_id_suffix):
            start_node = node
        if node.id.endswith(end_node_id_suffix):
            end_node = node

    assert start_node is not None and end_node is not None, \
        'Failed to find start/end nodes of the pre-processing block. The section of the transformation JSON ' \
        'configuration file related to "ObjectDetectionAPIPreprocessor2Replacement" transformation should be updated ' \
        'for this particular model.'
    allowed_ops = ['Sub', 'Mul', 'Div', 'Add']
    preprocessing_nodes = get_specific_ops_with_const_inputs(start_node, allowed_ops, False)
    trailing = False  # switch to apply newly created pre-processing nodes after/before start_node/end_node
    if len(preprocessing_nodes) == 0:
        preprocessing_nodes = get_specific_ops_with_const_inputs(end_node, allowed_ops, True)
        trailing = True

    # try to detect floating-point casting inside the body graph
    # that also needs to stay in the resulted graph
    casting = False
    cast_nodes = backward_bfs_for_operation(start_node, ['Cast'])
    if len(cast_nodes) == 1 and cast_nodes[0].dst_type == np.float32:
        casting = True

    return preprocessing_nodes, trailing, casting


""" 
Object Detection API models contain the sub-graph that performs some (not necessarily all) of the following tasks
(possibly in different order):
* Resizes image according to the constraints defined in the pipeline.config file.
* Applies mean and scale values.
* Pads the resized image to the size specified in the pipeline.config file.
This sub-graph is called "Preprocessor" in TF1 OD API models and early versions of the TF2 OD API models. Starting from
version 2.4 the block is called "map". The sub-graph has one output with the pre-processed input image and optionally
has a second output which contains either the original image size or the resized image size (before padding). When the
second output exists it is used to map predicted bounding boxes of the resized image to the original image coordinates.

Model Optimizer removes nodes performing image resize and padding, but keeps nodes applying mean and scale values. 
Historically, Model Optimizer didn't support converting TF sub-graphs into TensorIterator/Loop from TF 1 models so this
was the only option to convert the model and avoid dynamism which occurs when keep_aspect_ratio resizer is used. And the
user should resize the image the same way as it is implemented in the model before feeding the data to the Inference
Engine.

If the "keep_aspect_ratio" resizer with "pad_to_max_dimension" parameter equal to "true" is used and mean/scale
operations are applied before the resize like this:

input_tensor -> mean/scale -> resize -> pad -> ... 

then it is not allowed to remove the resize and padding operations and pre-process the input data before feeding the
model like this:

resized_padded_input_data -> mean/scale -> ...

because the output results will be different because mean/scale operations will be applied for padding area as well. So
the only option in this case is to remove all pre-processing operations from the model and expect that user perform them
before feeding the model.
"""


class ObjectDetectionAPIPreprocessorReplacement(FrontReplacementFromConfigFileSubGraph):
    """
    The transformation is triggered for the pre-processing block which resizes the input image and applies mean/scale
    values in the TF1 OD API models.
    """
    replacement_id = 'ObjectDetectionAPIPreprocessorReplacement'
    run_not_recursively = True

    def run_before(self):
        return [ObjectDetectionAPITransformationsFinish]

    def run_after(self):
        return [ObjectDetectionAPITransformationsStart]

    def nodes_to_remove(self, graph: Graph, match: SubgraphMatch):
        new_nodes_to_remove = match.matched_nodes_names()
        # do not remove nodes that perform input image scaling and mean value subtraction
        for node_to_keep in ('Preprocessor/sub', 'Preprocessor/sub/y', 'Preprocessor/mul', 'Preprocessor/mul/x'):
            if node_to_keep in new_nodes_to_remove:
                new_nodes_to_remove.remove(node_to_keep)
        return new_nodes_to_remove

    def is_preprocessing_applied_before_resize(self, to_float: Node, mul: Node, sub: Node):
        """
        The function checks if the output of 'to_float' operation is consumed by 'mul' or 'sub'. If this is true then
        the pre-processing (mean/scale) is applied before the image resize. The image resize was applied first in the
        original version of the TF OD API models, but in the recent versions it is applied after.

        :param to_float: the Cast node which converts the input tensor to Float
        :param mul: the Mul node (can be None)
        :param sub: the Sub node
        :return: the result of the check
        """
        assert sub is not None, 'The Sub node should not be None. Check the caller function.'
        if mul is not None:
            return any([port.node.id == mul.id for port in to_float.out_port(0).get_destinations()])
        else:
            return any([port.node.id == sub.id for port in to_float.out_port(0).get_destinations()])

    def generate_sub_graph(self, graph: Graph, match: SubgraphMatch):
        sub_node = match.output_node(0)[0]
        # sanity check whether this is really TF OD API model. The Sub operation always exists in TF1 OD API models
        # pre-processing sub-graph
        if sub_node.soft_get('op') != 'Sub':
            raise Error('The output op of the Preprocessor sub-graph is not of type "Sub". Looks like the topology is '
                        'not created with TensorFlow Object Detection API.')

        # identify the node performing scale (if it exists)
        mul_node = None
        if sub_node.in_port(0).get_source().node.soft_get('op') == 'Mul':
            log.info('There is image scaling node in the Preprocessor block.')
            mul_node = sub_node.in_port(0).get_source().node

        # update the model Parameter node shape based on MO command line parameters and values in the pipeline.config
        initial_input_node_name, placeholder_node = update_parameter_shape(graph, match)

        to_float_node = placeholder_node.out_port(0).get_destination().node
        # one more sanity check
        if to_float_node.soft_get('op') != 'Cast':
            raise Error('The output of the node "{}" is not Cast operation. Cannot apply transformation.'.format(
                initial_input_node_name))

        if self.is_preprocessing_applied_before_resize(to_float_node, mul_node, sub_node):
            # connect sub node directly to nodes which consume resized image
            resize_output_node_id = 'Preprocessor/map/TensorArrayStack/TensorArrayGatherV3'
            if resize_output_node_id not in graph.nodes:
                raise Error('There is no expected node "{}" in the graph.'.format(resize_output_node_id))
            resize_output = Node(graph, resize_output_node_id)
            for dst_port in resize_output.out_port(0).get_destinations():
                dst_port.get_connection().set_source(sub_node.out_port(0))
        else:
            # connect to_float_node directly with node performing scale on mean value subtraction
            if mul_node is None:
                to_float_node.out_port(0).connect(sub_node.in_port(0))
            else:
                to_float_node.out_port(0).connect(mul_node.in_port(1))

        log.error('The Preprocessor block has been removed. Only nodes performing mean value subtraction and scaling '
                  '(if applicable) are kept.', extra={'is_warning': True})
        # the pre-processing sub-graph is connected with the main graph, so there is no need to return new nodes mapping
        # dictionary
        return {}


class ObjectDetectionAPIPreprocessor2Replacement(FrontReplacementFromConfigFileGeneral):
    """
    The transformation is triggered for the pre-processing block which resizes the input image and applies mean/scale
    values in the TF2 OD API model. Only nodes related to applying mean/scaling values are kept.
    If the mean/scale values are applied before the resize and the pre-processing includes padding then mean/scale
    values are removed as well. Refer to the comments section before the ObjectDetectionAPIPreprocessorReplacement
    transformation.

    There are 6 possible cases:
    1. ... -> Scale -> Start -> Resize -> End -> ...
    2. ... -> Start -> Resize -> End -> Scale -> ...
    3. ... -> Start -> Resize -> End -> ...
    4. ... -> Start -> While (... -> Scale  -> Resize -> ...) -> End -> ...
    5. ... -> Start -> While (... -> Resize -> Scale -> ...) -> End -> ...
    6. ... -> Start -> While (... -> Resize -> ...) -> End -> ...

    Where:
    - "Start" - is the node name specified in the transformation configuration file
    - "End" - is the node name specified in the transformation configuration file
    - "Scale" - a node or a sequence of element-wise nodes like Mul, Add, Sub or Div with Const input
    - "While" (... nodes ... ) - a Loop operation with body nodes specified in parentheses
    - "Resize" - the Resize sub-graph being removed

    The transformation creates a new sub-graph of pre-processing nodes if in the original model it is inside the Loop,
    or keeps the existing one if they are in the main graph already.
    """
    replacement_id = 'ObjectDetectionAPIPreprocessor2Replacement'
    run_not_recursively = True

    def run_before(self):
        return [ObjectDetectionAPITransformationsFinish]

    def run_after(self):
        return [ObjectDetectionAPITransformationsStart]

    def transform_graph(self, graph: Graph, replacement_descriptions: dict):
        argv = graph.graph['cmd_params']
        if argv.tensorflow_object_detection_api_pipeline_config is None:
            raise Error(missing_param_error)

        pipeline_config = PipelineConfig(argv.tensorflow_object_detection_api_pipeline_config)
        pad_to_max_dimension = pipeline_config.get_param('pad_to_max_dimension')

        # update the model Parameter node shape based on MO command line parameters and values in the pipeline.config
        update_parameter_shape(graph, None)

        # NOTE: this transformation can be implemented as a "scope" or "points" transformation since we need to match
        # some sub-graph between specific nodes
        start_nodes = replacement_descriptions['start_nodes']
        end_nodes = replacement_descriptions['end_nodes']

        start_nodes = [node_id for node_id in start_nodes if node_id in graph.nodes]
        end_nodes = [node_id for node_id in end_nodes if node_id in graph.nodes]

        assert len(start_nodes) >= 1
        start_node = Node(graph, start_nodes[0])

        assert len(end_nodes) >= 1
        end_node = Node(graph, end_nodes[0])

        # determine nodes between specified input and output nodes to check if there is a Loop op among them
        sub_graph_node_ids = sub_graph_between_nodes(graph, start_nodes, end_nodes, include_control_flow=False,
                                                     allow_non_reachable_end_nodes=True)

        pre_processing_in_loop = False
        # If the pre-processing block contains Loop operation then mean and scale value should be obtained from it using
        # some pre-defined marker nodes existing for all pre-processing blocks.
        # If there is no Loop then pre-processing nodes are in the main graph and they should be obtained from it
        loop_nodes_ids = [node_id for node_id in sub_graph_node_ids if graph.nodes[node_id].get('op') == 'Loop']
        if len(loop_nodes_ids):
            assert len(loop_nodes_ids) == 1, 'There should be exactly one Loop node in the pre-processor block.'
            pre_processing_in_loop = True
            loop_node = Node(graph, loop_nodes_ids[0])
            body_graph = loop_node.body
            # we stick to the nodes with ids 'map/while/Preprocessor/unstack' and 'map/while/Preprocessor/stack' as they
            # "wrap" nodes performing image resize. The scale/mean values nodes are located strictly before or after
            # them
            pre_processing_ops, trailing, casting = get_preprocessing_ops(body_graph,
                                                                 'map/while/Preprocessor/unstack',
                                                                 'map/while/Preprocessor/stack')
        else:
            pre_processing_ops, trailing, casting = get_preprocessing_ops(graph, start_node.id, end_node.id)

        mean_scale_kept = True
        if len(pre_processing_ops):
            # if the pre-processing is applied before the resize then reverse them to be in the topological order
            if not trailing:
                pre_processing_ops = list(reversed(pre_processing_ops))

            if pre_processing_in_loop:  # case 4 and 5
                # build a sub-graph containing a sequence of pre_processing_ops if they came from the Loop
                new_preprocessing_ops = []

                # cast data before start pre-processing with mean/scale values
                if casting:
                    cast_node = Cast(graph, {'dst_type': np.float32}).create_node()
                    new_preprocessing_ops.append(cast_node)

                ops_mapping = {'Add': Add, 'Div': Div, 'Mul': Mul, 'Sub': Sub}
                for idx in range(len(pre_processing_ops)):
                    origin_node, const_port_ind, value = pre_processing_ops[idx]
                    new_node = create_op_with_const_inputs(graph, ops_mapping[origin_node.op], {const_port_ind: value})
                    if len(new_preprocessing_ops):
                        new_node.in_port(1 - const_port_ind).connect(new_preprocessing_ops[-1].out_port(0))
                    new_preprocessing_ops.append(new_node)

                # replace sub-graph between start and end nodes (including them) with new_preprocessing_ops nodes
                end_node.out_port(0).get_connection().set_source(new_preprocessing_ops[-1].out_port(0))
                start_node.in_port(0).get_connection().set_destination(
                    new_preprocessing_ops[0].in_port(int(new_preprocessing_ops[0].is_in_port_connected(0))))
            else:
                if trailing:  # case 2
                    # change output of the end_node to be produced with the start node producer
                    source_port = start_node.in_port(0).get_source()
                    source_port.disconnect()
                    end_node.out_port(0).get_connection().set_source(source_port)
                else:  # case 1
                    # if padding is specified then need to remove mean/scale as well. Refer to the transformation
                    # comments for more details
                    if pad_to_max_dimension:
                        # change output of the end_node to be produced with the node producing data for the first
                        # preprocessing op
                        mean_scale_kept = False
                        first_pre_processing_node = pre_processing_ops[0][0]
                        consumer_port = first_pre_processing_node.in_port(int(not pre_processing_ops[0][1]))
                        end_node.out_port(0).get_connection().set_source(consumer_port.get_connection().get_source())
                    else:
                        # change output of the end_node to be produced with the last preprocessing op
                        end_node.out_port(0).get_connection().set_source(pre_processing_ops[-1][0].out_port(0))
                        start_node.in_port(0).disconnect()
        else:  # simply remove the nodes in between start_node and end_node (including them). Case 3 and 6
            end_node.out_port(0).get_connection().set_source(start_node.in_port(0).get_source())

        if mean_scale_kept:
            log.error('The pre-processing block has been removed. Only nodes performing mean value subtraction and '
                      'scaling (if applicable) are kept. It is necessary to resize an input image using the same '
                      'algorithm as in the original model before feeding it to the OpenVINO.',
                      extra={'is_warning': True})
        else:
            log.error('The Preprocessor block has been removed including mean value subtraction and scaling (if '
                      'applicable). It is necessary to resize, scale and pad an input image using the same algorithm '
                      'as in the original model before feeding it to the OpenVINO.', extra={'is_warning': True})


class ObjectDetectionAPIDetectionOutputReplacement(FrontReplacementFromConfigFileSubGraph):
    """
    Replaces the sub-graph that is equal to the DetectionOutput layer from OpenVINO (similarly to the
    ObjectDetectionAPISSDPostprocessorReplacement). This transformation is used for Faster R-CNN, R-FCN and Mask R-CNN
    topologies conversion.
    Refer to the code for more details.
    """
    replacement_id = 'ObjectDetectionAPIDetectionOutputReplacement'
    run_not_recursively = True

    def run_before(self):
        return [ObjectDetectionAPIMaskRCNNROIPoolingSecondReplacement]

    def run_after(self):
        return [ObjectDetectionAPIProposalReplacement]

    def nodes_to_remove(self, graph: Graph, match: SubgraphMatch):
        new_nodes_to_remove = match.matched_nodes_names().copy()
        outputs = ['detection_boxes', 'detection_scores', 'num_detections']
        for output in outputs:
            if output in graph.nodes:
                children = Node(graph, output).out_nodes()
                if len(children) != 1:
                    log.warning('Output {} has {} children. It should have only one output: with op==`Result`'
                                ''.format(output, len(children)))
                elif children[list(children.keys())[0]].op == 'Result':
                    new_nodes_to_remove.append(children[list(children.keys())[0]].id)
        new_nodes_to_remove.extend(outputs)
        return new_nodes_to_remove

    def output_edges_match(self, graph: Graph, match: SubgraphMatch, new_sub_graph: dict):
        # the DetectionOutput in OV produces single tensor, but in TF it produces four tensors, so we need to create
        # only one output edge match
        if match.outputs_count() >= 1:
            return {match.output_node(0)[0].id: new_sub_graph['detection_output_node'].id}
        else:
            return {list(graph.graph.get("packed_outputs").keys())[0]: new_sub_graph['detection_output_node'].id}

    def generate_sub_graph(self, graph: Graph, match: SubgraphMatch):
        argv = graph.graph['cmd_params']
        if argv.tensorflow_object_detection_api_pipeline_config is None:
            raise Error(missing_param_error)
        pipeline_config = PipelineConfig(argv.tensorflow_object_detection_api_pipeline_config)
        custom_attributes = match.custom_replacement_desc.custom_attributes

        num_classes = _value_or_raise(match, pipeline_config, 'num_classes')
        max_proposals = _value_or_raise(match, pipeline_config, 'first_stage_max_proposals')
        activation_function = _value_or_raise(match, pipeline_config, 'postprocessing_score_converter')

        activation_conf_node = add_activation_function_after_node(graph, match.single_input_node(1)[0].in_node(0),
                                                                  activation_function)

        # OV DetectionOutput operation consumes flattened tensors so need add a Reshape layer.
        # The batch value of the input tensor is not equal to the batch of the topology, so it is not possible to use
        # "0" value in the Reshape layer attribute to refer to the batch size, but we know how to
        # calculate the second dimension so the batch value will be deduced from it with help of "-1".
        reshape_conf_node = create_op_node_with_second_input(graph, Reshape,
                                                             int64_array([-1, (num_classes + 1) * max_proposals]),
                                                             dict(name='do_reshape_conf'), activation_conf_node)
        mark_as_correct_data_layout(reshape_conf_node)

        # We looking for first not Reshape-typed node before match.single_input_node(0)[0].in_node(0).
        # And add reshape_offsets node after this first not Reshape-typed node to avoid issues with Reshape-like
        # operations which may trigger insert of Transpose operations before/after them
        current_node = skip_nodes_by_condition(match.single_input_node(0)[0].in_node(0),
                                               lambda x: x['kind'] == 'op' and x.has_and_set('reinterp_shape'))

        # if share_box_across_classes=1 then the same set of bounding boxes shape offsets is used for all classes,
        # otherwise per-class set of shape offsets is used and we need to use appropriate Reshape output shape
        share_box_across_classes = _value_or_raise(match, pipeline_config, 'share_box_across_classes')
        if share_box_across_classes:
            reshape_offsets_shape = int64_array([-1, 1, 1, 4])
        else:
            reshape_offsets_shape = int64_array([-1, num_classes, 1, 4])
        reshape_offsets = create_op_node_with_second_input(graph, Reshape, reshape_offsets_shape,
                                                           dict(name='reshape_loc'), current_node)
        mark_as_correct_data_layout(reshape_offsets)

        if share_box_across_classes:
            offsets = reshape_offsets
        else:
            # TF produces shape offsets tensor without boxes corresponding to "background" class
            # OpenVINO DetectionOutput layer requires "background" class data be included so we generate them
            offsets = add_fake_background_loc(graph, reshape_offsets)
            PermuteAttrs.set_permutation(reshape_offsets, offsets, None)

        # reshape offsets tensor to 2D so it could be multiplied with variances
        reshape_offsets_2d = create_op_node_with_second_input(graph, Reshape, int64_array([-1, 4]),
                                                              dict(name='reshape_locs_2d'), offsets)
        mark_as_correct_data_layout(reshape_offsets_2d)

        # multiply bounding boxes shape offsets with variances as it is expected when variance_encoded_in_target=1 for
        # the DetectionOutput operation
        variances = Const(graph, dict(value=_variance_from_pipeline_config(pipeline_config))).create_node([])
        scaled_offsets = Mul(graph, dict()).create_node([reshape_offsets_2d, variances], dict(name='scale_locs'))

        # there are Convolution/MatMul nodes before the post-processing block in all models except RFCN. So for most of
        # the models we can just update Convolution/MatMul weights to perform swapping of coordinates. But for the RFCN
        # models we use approach with adding special Convolution node which perform the same swap. Previously we used a
        # dedicated parameter in the transformation config but now it is not needed and the get this information from
        # the model automatically by performing graph traversal until CropAndResize (RFCN case) or Conv/MatMul nodes are
        # found
        if 'coordinates_swap_method' in custom_attributes:
            log.error('The "coordinates_swap_method" parameter is not needed anymore. Consider removing it from the '
                      '"ObjectDetectionAPIDetectionOutputReplacement" transformation custom attributes.',
                      extra={'is_warning': True})
        matmul_or_conv_nodes = backward_bfs_for_operation(scaled_offsets, ['MatMul', 'Conv2D'], ['ShapeOf',
                                                                                                 'CropAndResize'])
        if len(matmul_or_conv_nodes) == 0:
            swapped_offsets = add_convolution_to_swap_xy_coordinates(graph, scaled_offsets, 4)
            flattened_offsets = Reshape(graph, dict(name='do_reshape_locs')).create_node([swapped_offsets])
        else:
            swap_weights_xy(graph, matmul_or_conv_nodes)
            flattened_offsets = Reshape(graph, dict(name='do_reshape_locs')).create_node([scaled_offsets])

        # OV DetectionOutput layer consumes flattened tensors so need add a Reshape layer.
        # The batch value of the input tensor is not equal to the batch of the topology, so it is not possible to use
        # "0" value in the Reshape layer attribute to refer to the batch size, but we know how to
        # calculate the second dimension so the batch value will be deduced from it with help of "-1".
        if share_box_across_classes:
            reshape_shape = int64_array([-1, max_proposals * 4])
        else:
            reshape_shape = int64_array([-1, (num_classes + 1) * max_proposals * 4])
        Const(graph, {'value': reshape_shape, 'name': flattened_offsets.name + '/Dim'}).create_node().out_port(0).\
            connect(flattened_offsets.in_port(1))
        mark_as_correct_data_layout(flattened_offsets)

        # find Proposal output which has the data layout as in TF: YXYX coordinates without batch indices.
        proposal_nodes_ids = [node_id for node_id, attrs in graph.nodes(data=True)
                              if 'name' in attrs and attrs['name'] == 'crop_proposals']
        if len(proposal_nodes_ids) != 1:
            raise Error("Found the following nodes '{}' with name 'crop_proposals' but there should be exactly 1. "
                        "Looks like ObjectDetectionAPIProposalReplacement transformation didn't work."
                        "".format(proposal_nodes_ids))
        proposal = Node(graph, proposal_nodes_ids[0])

        # Need to swap proposals coordinates before passing them to the DetectionOutput for the RFCN topologies
        if len(matmul_or_conv_nodes) == 0:
            proposal = add_convolution_to_swap_xy_coordinates(graph, proposal, 4)

        # reshape priors boxes as Detection Output expects
        reshape_priors = create_op_node_with_second_input(graph, Reshape, int64_array([-1, 1, max_proposals * 4]),
                                                          dict(name='DetectionOutput_reshape_priors_'), proposal)
        mark_as_correct_data_layout(reshape_priors)

        detection_output_op = DetectionOutput(graph, {})
        for key in ('clip_before_nms', 'clip_after_nms'):
            if key in match.custom_replacement_desc.custom_attributes:
                detection_output_op.attrs[key] = int(match.custom_replacement_desc.custom_attributes[key])

        detection_output = detection_output_op.create_node([flattened_offsets, reshape_conf_node, reshape_priors], dict(
            name=detection_output_op.attrs['type'],
            share_location=int(share_box_across_classes),
            variance_encoded_in_target=1,
            background_label_id=int(custom_attributes.get('background_label_id', 0)),
            code_type='caffe.PriorBoxParameter.CENTER_SIZE',
            pad_mode='caffe.ResizeParameter.CONSTANT',
            resize_mode='caffe.ResizeParameter.WARP',
            confidence_threshold=_value_or_raise(match, pipeline_config, 'postprocessing_score_threshold'),
            top_k=_value_or_raise(match, pipeline_config, 'postprocessing_max_detections_per_class'),
            keep_top_k=_value_or_raise(match, pipeline_config, 'postprocessing_max_total_detections'),
            nms_threshold=_value_or_raise(match, pipeline_config, 'postprocessing_iou_threshold')))
        # sets specific name to the node so we can find it in other transformations
        detection_output.name = 'detection_output'

        # when the use_matmul_crop_and_resize = True then the prior boxes were not swapped and we need to swap them from
        # YXYX to XYXY before passing to the DetectionOutput operation
        if pipeline_config.get_param('use_matmul_crop_and_resize'):
            insert_weights_swap_xy_sub_graph(graph, detection_output.in_port(2).get_connection())

        # create Result since after the transformation other Results are removed
        Result(graph, dict(name='do_OutputOp')).create_node([detection_output])

        log.error('The graph output nodes have been replaced with a single layer of type "DetectionOutput". Refer to '
                  'the operation set specification documentation for more information about the operation.',
                  extra={'is_warning': True})
        return {'detection_output_node': detection_output}


class ObjectDetectionAPIMaskRCNNROIPoolingSecondReplacement(FrontReplacementFromConfigFileSubGraph):
    """
    There are two TensorFlow CropAndResize (corresponding to OpenVINO ROIPooling with bilinear interpolation
    mode) operations in the Mask-RCNN model. The second CropAndResize gets bounding boxes coordinates as input from the
    part of the model which is replaced with the DetectionOutput operation using the transformation
    ObjectDetectionAPIDetectionOutputReplacement. DetectionOutput operation produces tensor with 7-element tuples
    [batch_id, class_id, confidence, x_1, y_1, x_2, y_2]. The ROIPooling operation expects input defining bounding boxes
    with the following format [batch_id, x_1, y_1, x_2, y_2]. The ObjectDetectionAPIMaskRCNNROIPoolingSecondReplacement
    transformation inserts ROIPooling operation instead of the CropAndResize and crops slices of data from the
    DetectionOutput operation and concatenates them to produce a tensor with correct content.
    """
    replacement_id = 'ObjectDetectionAPIMaskRCNNROIPoolingSecondReplacement'
    run_not_recursively = True

    def run_before(self):
        return [ObjectDetectionAPITransformationsFinish]

    def run_after(self):
        return [ObjectDetectionAPIProposalReplacement]

    def output_edges_match(self, graph: Graph, match: SubgraphMatch, new_sub_graph: dict):
        return {match.output_node(0)[0].id: new_sub_graph['roi_pooling_node'].id}

    def generate_sub_graph(self, graph: Graph, match: SubgraphMatch):
        argv = graph.graph['cmd_params']
        if argv.tensorflow_object_detection_api_pipeline_config is None:
            raise Error(missing_param_error)
        pipeline_config = PipelineConfig(argv.tensorflow_object_detection_api_pipeline_config)

        # the output spatial dimensions of the ROIPooling operation are defined in the pipeline.config
        roi_pool_size = _value_or_raise(match, pipeline_config, 'initial_crop_size')

        # find the DetectionOutput operation by name to get tensor with information about bounding boxes from it.
        # the layout of bounding boxes is XYXY already, so no need to swap them
        detection_output_nodes_ids = [node_id for node_id, attrs in graph.nodes(data=True)
                                      if 'name' in attrs and attrs['name'] == 'detection_output']
        if len(detection_output_nodes_ids) != 1:
            raise Error("Found the following nodes '{}' with name 'detection_output' but there should be exactly 1.".
                        format(detection_output_nodes_ids))
        detection_output = Node(graph, detection_output_nodes_ids[0])
        do_outputs = [port.node for port in detection_output.out_port(0).get_destinations() if port.node.op == 'Result']
        if len(do_outputs) == 1:
            graph.remove_node(do_outputs[0].id)

        # add reshape of Detection Output so it can be an output of the topology.
        # this looks like some legacy not relevant constraint anymore
        flatten_do = create_op_node_with_second_input(graph, Reshape, int64_array([-1, 7]), dict(name='reshape_do_2d'),
                                                      detection_output)
        mark_as_correct_data_layout(flatten_do)

        # adds "Result" node so this output is returned by OV by default for the backward compatibility
        do_result = Result(graph, dict(name='do_reshaped_OutputOp')).create_node([flatten_do])

        # add attribute 'output_sort_order' so it will be used as a key to sort output nodes before generation of IR
        do_result.in_edge()['data_attrs'].append('output_sort_order')
        do_result.in_edge()['output_sort_order'] = [('detection_boxes', 0)]

        # creates two Crop operations which get input from the DetectionOutput, cuts off slices of data with class ids
        # and probabilities and produces a tensor with batch ids and bounding boxes only (as it is expected by the
        # ROIPooling operation)
        batch_ids = Crop(graph, dict(axis=int64_array([1]), offset=int64_array([0]), dim=int64_array([1]))).create_node(
            [flatten_do], dict(name='crop_do_batch_ids'))
        coords = Crop(graph, dict(axis=int64_array([1]), offset=int64_array([3]), dim=int64_array([4]))).create_node(
            [flatten_do], dict(name='crop_do_coords'))
        batch_and_coords = Concat(graph, dict(axis=1)).create_node([batch_ids, coords], dict(name='batch_and_coords'))

        roi_pooling = ROIPooling(graph, dict(method="bilinear", spatial_scale=1, pooled_h=roi_pool_size,
                                             pooled_w=roi_pool_size)).create_node(
            [match.single_input_node(0)[0].in_node(), batch_and_coords], dict(name='ROI_pooling_2'))
        return {'roi_pooling_node': roi_pooling}


class ObjectDetectionAPIMaskRCNNSigmoidReplacement(FrontReplacementFromConfigFileGeneral):
    """
    The transformation is used to convert Mask R-CNN topologies only.

    The post-processing part of Mask-RCNN models is to select masks from the output tensor which correspond to bounding
    boxes with probability exceeding specific threshold. The final step of the post-processing is to apply Sigmoid
    activation function to the tensor with selected masks so the values become in range [0, 1]. The post-processing part
    of the model is not supported so it is removed using the transformation ObjectDetectionAPIOutputReplacement.
    This transformation adds back the activation function to the end of the network producing masks tensors. So the
    post-processing with selecting masks corresponding to bounding boxes with high probabilities should be implemented
    in the application.
    """
    replacement_id = 'ObjectDetectionAPIMaskRCNNSigmoidReplacement'
    run_not_recursively = True

    def run_before(self):
        return [ObjectDetectionAPITransformationsFinish]

    def run_after(self):
        return [ObjectDetectionAPIMaskRCNNROIPoolingSecondReplacement]

    def transform_graph(self, graph: Graph, replacement_descriptions):
        # there could be multiple Result nodes in the graph. We identify the one containing masks data using the node
        # name prefix
        masks_node_prefix_name = replacement_descriptions.get('masks_node_prefix_name', 'SecondStageBoxPredictor')
        op_outputs = graph.get_op_nodes(op='Result')
        for op_output in op_outputs:
            last_node = op_output.in_port(0).get_source().node
            if last_node.name.startswith(masks_node_prefix_name):
                sigmoid_node = Sigmoid(graph, dict(name='masks')).create_node()
                op_output.in_port(0).get_connection().insert_node(sigmoid_node)

                # the line below is needed to keep layout as is, istead of default NCHW->NHWC changing
                sigmoid_node['nchw_layout'] = True

                # adding op name to tensor names list is needed for compatiblity with old api configs
                op_output.in_port(0).get_connection().get_source().add_tensor_names([sigmoid_node['name']])

        log.error('The predicted masks are produced by the "masks" layer for each bounding box generated with a '
                  '"detection_output" operation.\n Refer to operation specification in the documentation for the '
                  'information about the DetectionOutput operation output data interpretation.\nThe model can be '
                  'inferred using the dedicated demo "mask_rcnn_demo" from the OpenVINO Open Model Zoo.',
                  extra={'is_warning': True})


class ObjectDetectionAPIProposalReplacement(FrontReplacementFromConfigFileSubGraph):
    """
    The outputs of the Region Proposal Network which produces shape offsets and probabilities whether anchors contain
    object or not is fed to the part of the model which decodes bounding boxes and performs non-maximum suppression.
    There are two operations in the OpenVINO which can perform such calculations: Proposal and DetectionOutput.
    Historically, the Proposal operation was inserted by this transformation, but now a DetectionOutput can be inserted
    instead if the "operation_to_add" parameter in the JSON configuration file is set to "DetectionOutput". There was a
    model for which inserting DetectionOutput instead of Proposal operation results in generation more accurate results.
    Another reason why Proposal operation is not preferable is that it requires addition model input which defines
    original image size and special scale value (refer to the operation specification for more details). So even though
    the original TensorFlow model has one input (actual image), the generated IR contains two inputs (actual image and
    a special input for the Proposal operation). It is not possible to switch to inserting DetectionOutput operation
    by default because it is not backward compatible change and some customer script may start to fail since one input
    disappears.
    Refer to the code for details on the conversion process and operations inserted.
    """
    replacement_id = 'ObjectDetectionAPIProposalReplacement'
    run_not_recursively = True
    matched_input_nodes_to_keep = 2  # number of matched input nodes to keep

    def run_after(self):
        return [ObjectDetectionAPIPreprocessorReplacement, ObjectDetectionAPIPreprocessor2Replacement]

    def run_before(self):
        return [ObjectDetectionAPITransformationsFinish]

    def output_edges_match(self, graph: Graph, match: SubgraphMatch, new_sub_graph: dict):
        return {match.output_node(0)[0].id: new_sub_graph['proposal_node'].id}

    def nodes_to_remove(self, graph: Graph, match: SubgraphMatch):
        new_list = match.matched_nodes_names().copy()
        # do not remove nodes that produce box predictions and class predictions and optionally generated anchors
        for port in range(self.matched_input_nodes_to_keep):
            new_list.remove(match.single_input_node(port)[0].id)
        return new_list

    def generate_sub_graph(self, graph: Graph, match: SubgraphMatch):
        argv = graph.graph['cmd_params']
        if argv.tensorflow_object_detection_api_pipeline_config is None:
            raise Error(missing_param_error)
        pipeline_config = PipelineConfig(argv.tensorflow_object_detection_api_pipeline_config)

        # the transformation configuration file specifies what operations should be included with this transformation
        if match.custom_replacement_desc.custom_attributes.get('operation_to_add', 'Proposal') == 'DetectionOutput':
            self.matched_input_nodes_to_keep = 3  # keep the third input with prior boxes (anchors)
            return self.insert_detection_output_instead_of_proposal(graph, match, pipeline_config)

        max_proposals = _value_or_raise(match, pipeline_config, 'first_stage_max_proposals')
        proposal_ratios = _value_or_raise(match, pipeline_config, 'anchor_generator_aspect_ratios')
        proposal_scales = _value_or_raise(match, pipeline_config, 'anchor_generator_scales')
        anchors_count = len(proposal_ratios) * len(proposal_scales)

        # Find Convolution/MatMul node that produces classes confidence
        class_conf = backward_bfs_for_operation(match.single_input_node(1)[0], ['Add'])[0]

        # size of 'C' dimension of the tensor with class predictions is equal to base_anchors_count * 2, where 2
        # corresponds to a number of classes (background and foreground) and base_anchors_count is equal to number of
        # anchors applied to each position of 'H' and 'W' dimensions. Therefore, there are H * W * base_anchors_count
        # bounding boxes. OpenVINO Proposal operation interprets the input tensor as a tensor
        # [batch, 2 * base_anchors_count, H, W] but in TensorFlow model it is calculated as
        # [batch, base_anchors_count, H, W] (after NHWC->NCHW layout conversion), so it is necessary to decompose the
        # 'C' dimension into base_anchors_count and 2 and swap these two dimensions
        reshape_class_conf = create_op_node_with_second_input(graph, Reshape, int64_array([0, anchors_count, 2, -1]),
                                                              dict(name='predictions/Reshape'))
        class_conf.insert_node_after(reshape_class_conf, 0)
        mark_as_correct_data_layout(reshape_class_conf)

        # the part of the sub-graph being removed contains the SoftMax operation, so here we insert it back
        softmax_conf_op = Softmax(graph, dict(axis=2, nchw_layout=True, name=reshape_class_conf.id + '/Softmax'))
        softmax_conf = softmax_conf_op.create_node([reshape_class_conf])

        order_const = Const(graph, dict(value=int64_array([0, 2, 1, 3]),
                                        name=softmax_conf.name + '/TransposeOrder')).create_node()
        permute_reshape_softmax_op = Transpose(graph, dict())
        permute_reshape_softmax = permute_reshape_softmax_op.create_node([softmax_conf, order_const], dict(
            name=softmax_conf.name + '/Transpose'))
        mark_input_as_in_correct_layout(permute_reshape_softmax, 1)
        mark_output_as_in_correct_layout(permute_reshape_softmax, 0)

        initial_shape = Shape(graph, dict(name=class_conf.id + '/Shape')).create_node([class_conf])

        reshape_conf_initial = Reshape(graph, dict(name='Reshape_Transpose_Class')).create_node(
            [permute_reshape_softmax, initial_shape])
        mark_input_as_in_correct_layout(reshape_conf_initial, 0)
        mark_output_as_in_correct_layout(reshape_conf_initial, 0)

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
                      'width stride "{}". The detection results will be inaccurate.'
                      ''.format(anchor_generator_height_stride, anchor_generator_width_stride))
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

        bboxes_offsets = backward_bfs_for_operation(match.single_input_node(0)[0], ['Add'])[0]

        # creates input to store input image height, width and scales (usually 1.0s) which is a mandatory input to the
        # Proposal operation. The batch size for this input is fixed because it is allowed to pass images of the same
        # size only as input
        im_info = Parameter(graph, dict(shape=int64_array([1, 3]), fixed_batch=True)).create_node(
            [], dict(name='image_info'))

        proposal = proposal_op.create_node([reshape_conf_initial, bboxes_offsets, im_info], dict(name='proposals'))
        return {'proposal_node': ObjectDetectionAPIProposalReplacement.ie_to_tf_proposals(graph, proposal, match,
                                                                                          pipeline_config,
                                                                                          max_proposals)}

    @staticmethod
    def insert_detection_output_instead_of_proposal(graph: Graph, match: SubgraphMatch,
                                                    pipeline_config: PipelineConfig):
        """
        The function inserts DetectionOutput operation instead of Proposal operation which may result in an increase of
        the accuracy for some models. The function is enabled with the custom attribute "operation_to_insert" with
        value "DetectionOutput" in the transformation configuration file section for the
        "ObjectDetectionAPIProposalReplacement" transformation.

        :param graph: the graph to operate on
        :param match: the object containing information about the matched sub-graph
        :param pipeline_config: object containing information from the pipeline.config file of the model
        :return: the dictionary with mapping information needed for other transformations
        """
        max_proposals = _value_or_raise(match, pipeline_config, 'first_stage_max_proposals')

        # Convolution/matmul node that produces classes confidence
        # Transpose result of the tensor with classes confidences so it will be in a correct layout for Softmax
        class_conf_nodes = backward_bfs_for_operation(match.single_input_node(1)[0], ['Add'])
        assert len(class_conf_nodes) >= 1, 'Expected to find nodes of type "Add" starting from the node "{}" in ' \
                                           'backward direction'.format(match.single_input_node(1)[0].id)
        class_conf = class_conf_nodes[0]

        # prepare input with class confidences. The DetectionOutput operation which will consume this tensor as a
        # second input expects probabilities to be normalized with SoftMax operation per each bounding box class. In
        # order to do this we first reshape the tensor so the last dimension contains probability for 2 classes
        # (background and foreground) for each bounding box. Before feeding this tensor to the DO operation the tensor
        # is flattened to the shape [num_batches, num_classes * num_bounding_boxes]
        reshape_conf = create_op_node_with_second_input(graph, Reshape, int64_array([0, -1, 2]),
                                                        dict(name='predictions/Reshape'))
        # transpose from NCHW to NHWC will be inserted as input to the Reshape automatically. This is expected
        class_conf.out_port(0).disconnect()
        class_conf.out_port(0).connect(reshape_conf.in_port(0))
        softmax_conf = Softmax(graph, dict(axis=2, name=reshape_conf.id + '/Softmax')).create_node([reshape_conf])
        flattened_conf = create_op_node_with_second_input(graph, Reshape, int64_array([0, -1]),
                                                          dict(name=softmax_conf.name + '/Flatten'), softmax_conf)
        # prepare input with bounding boxes shape offsets
        offsets = backward_bfs_for_operation(match.single_input_node(0)[0], ['Add'])[0]
        flatten_offsets = create_op_node_with_second_input(graph, Reshape, int64_array([0, -1]),
                                                           dict(name=offsets.soft_get('name', offsets.id) + '/Flatten'),
                                                           offsets)

        # TensorFlow produces anchor boxes in absolute coordinates in YXYX order. Need to normalize them to [0, 1]
        # interval and append a tensor with variances. Refer to the ObjectDetectionAPISSDPostprocessorReplacement
        # transformation comments about variances. The YXYX->XYXY order change will be performed with the output of the
        # inserted DetectionOutput operation
        yxyx_anchors = match.single_input_node(2)[0]

        # get the input image height and width to divide the anchors values by it
        initial_input_node_name = 'input_tensor' if 'input_tensor' in graph.nodes else 'image_tensor'
        if initial_input_node_name not in graph.nodes():
            raise Error('Input node "{}" of the graph is not found. Do not run the Model Optimizer with '
                        '"--input" command line parameter.'.format(initial_input_node_name))
        parameter_node = Node(graph, initial_input_node_name)

        input_shape = Shape(graph, {'name': parameter_node.name}).create_node([parameter_node])
        input_image_hw = node_to_get_shape_value_of_indices(input_shape, [1, 2])  # NHWC layout
        hwhw = create_op_with_const_inputs(graph, Tile, {1: int64_array([2])}, {'name': 'image_hwhw'}, input_image_hw)

        hwhw_float = Cast(graph, {'dst_type': np.float32}).create_node([hwhw])
        scaled_anchors = Div(graph, {'name': 'scaled_anchors'}).create_node([yxyx_anchors, hwhw_float])

        flattened_anchors = create_op_with_const_inputs(graph, Reshape, {1: int64_array([1, 1, -1])},
                                                        {'name': 'flattened_anchors'}, scaled_anchors)
        cropped_anchors = AttributedClamp(graph, {'min': 0.0, 'max': 1.0, 'name': 'clamped_yxyx',
                                                  'nchw_layout': True}).create_node([flattened_anchors])
        # the input tensor "scaled_anchors" for the "flattened_anchors" may be 4D. In order to avoid inserting Transpose
        # operation mark the "flattened_anchors" with the correct data layout
        mark_as_correct_data_layout(flattened_anchors)

        # create tensor of shape [4] with variance values which then are tiled by the number of boxes which is obtained
        # from the 'yxyx_anchors' node
        variances = Const(graph, {'value': _variance_from_pipeline_config(pipeline_config)}).create_node()

        anchors_shape = Shape(graph, {'name': 'anchors_shape'}).create_node([yxyx_anchors])
        anchors_count = node_to_get_shape_value_of_indices(anchors_shape, [0])
        tiled_variances = Tile(graph, {'name': 'tiled_variances'}).create_node([variances, anchors_count])
        reshaped_tiled_variances = create_op_with_const_inputs(graph, Reshape, {1: int64_array([1, 1, -1])},
                                                               {'name': 'flattened_variances'}, tiled_variances)

        # now we can merge actual anchors coordinates with a tensor with variances as it is expected by the
        # DetectionOutput operation
        duplicate_anchors = Concat(graph, {'axis': 1, 'name': 'anchors_with_variances'}).create_node(
            [cropped_anchors, reshaped_tiled_variances])

        do = DetectionOutput(graph,
                             {'background_label_id': 0,
                              'clip_after_nms': True,
                              'clip_before_nms': False,
                              'code_type': 'caffe.PriorBoxParameter.CENTER_SIZE',
                              'confidence_threshold': 0.0,
                              'decrease_label_id': False,
                              'input_height': 1,
                              'input_width': 1,
                              'keep_top_k': max_proposals,
                              'normalized': True,
                              'objectness_score': 0,
                              'share_location': True,
                              'top_k': 6000,
                              'variance_encoded_in_target': False,
                              'nms_threshold': _value_or_raise(match, pipeline_config, 'first_stage_nms_iou_threshold'),
                              'name': 'first_do',
                              }).create_node([flatten_offsets, flattened_conf, duplicate_anchors])
        # DetectionOutput output tensor has YXYX box coordinates order
        # switch to 3D to avoid issues that part of the model with 4D shapes should be inferred in NCHW layout
        do_3d = create_op_with_const_inputs(graph, Squeeze, {1: int64_array(0)}, {'name': do.name + '/SqueezeDO'}, do)
        mark_as_correct_data_layout(do_3d)

        # DetectionOutput output tensor produces a tensor of tuples with the following 7 elements:
        # [batch_id, class_id, confidence, x1, y1, x2, y2]. Here we split the DetectionOutput result into the 7
        # tensors with each of these elements for predictions. Then we crop predicted box coordinates (scaled) to be
        # within [0, 1] range (as it is predicted in the TF model) and then combine tensors back to the Proposal
        # operation output format: [batch_id, x1, y1, x2, y2].
        do_split = create_op_node_with_second_input(graph, Split, int64_array(2), {'num_splits': 7,
                                                                                   'name': do.name + '/Split'}, do_3d)

        coords = Concat(graph, {'axis': -1, 'in_ports_count': 4, 'name': do_split.name + '/coords'}).create_node()
        # concat bounding boxes with the same order (XYXY) as Proposal produces
        for port_idx in range(4):
            do_split.out_port(3 + port_idx).connect(coords.in_port(port_idx))

        clamped_coords = AttributedClamp(graph, {'min': 0.0, 'max': 1.0, 'name': 'clamped_xyxy'}).create_node([coords])

        # prepare final proposal boxes [batch_id, x1, y1, x2, y2]
        proposal_node = Concat(graph, {'axis': -1, 'in_ports_count': 2, 'name': 'proposals'}).create_node()
        do_split.out_port(0).connect(proposal_node.in_port(0))
        clamped_coords.out_port(0).connect(proposal_node.in_port(1))
        return {'proposal_node': ObjectDetectionAPIProposalReplacement.ie_to_tf_proposals(graph, proposal_node, match,
                                                                                          pipeline_config,
                                                                                          max_proposals)}

    @staticmethod
    def ie_to_tf_proposals(graph: Graph, proposal: Node, match: SubgraphMatch, pipeline_config: PipelineConfig,
                           max_proposals: int):
        """
        Builds a graph which converts the proposals data in OV format to the format of TensorFlow. This includes
        cropping the OV output of format [batch, x1, y1, x2, y2] to simply [x1, y1, x2, y2] and reshaping tensor to an
        appropriate shape. Swapping of the Proposal output is performed when necessary.

        :param graph: the graph to operate on
        :param proposal: the node producing OV proposals
        :param match: the object containing information about matched sub-graph
        :param pipeline_config: object containing information from the pipeline.config file of the model
        :param max_proposals: maximum number of proposal boxes. Needed for the reshaping of the tensor
        :return: the node producing output in the TF format.
        """
        # models with use_matmul_crop_and_resize = True should not swap order of elements (YX to XY) after the Proposal
        # because the TF output has XYXY layout originally.
        # Also old version of RFCN model (1.9) does not require proposal swap since the output has proper layout
        # already. The swap is controlled with the 'do_not_swap_proposals' parameter from the transformation file
        swap_proposals = not match.custom_replacement_desc.custom_attributes.get('do_not_swap_proposals', False) and \
                         not pipeline_config.get_param('use_matmul_crop_and_resize')
        if swap_proposals:
            proposal = add_convolution_to_swap_xy_coordinates(graph, proposal, 5)

        # the "reshape_swap_proposals_2d" is used in the ObjectDetectionAPIPSROIPoolingReplacement transformation. It
        # is important that this input may be swapped several lines above
        proposal_reshape_2d = create_op_node_with_second_input(graph, Reshape, int64_array([-1, 5]),
                                                               dict(name="reshape_swap_proposals_2d"), proposal)
        mark_input_as_in_correct_layout(proposal_reshape_2d, 0)

        # Find closest CropAndResize in topological order
        start_node = match.single_input_node(0)[0]
        crop_and_resize_nodes_ids = [node.id for node in graph.pseudo_topological_sort_with_start_node(start_node) if
                                     graph.nodes[node.id]['op'] == 'CropAndResize']

        if len(crop_and_resize_nodes_ids) != 0 and swap_proposals:
            # feed the CropAndResize node with a correct boxes information produced with the Proposal layer
            # find the first CropAndResize node in the BFS order. This is needed in the case when we already swapped
            # box coordinates data after the Proposal node
            crop_and_resize_node = Node(graph, crop_and_resize_nodes_ids[0])
            # set a marker that an input with box coordinates has been pre-processed so the CropAndResizeReplacement
            # transform doesn't try to merge the second and the third inputs
            crop_and_resize_node['inputs_preprocessed'] = True
            crop_and_resize_node.in_port(1).disconnect()
            proposal_reshape_2d.out_port(0).connect(crop_and_resize_node.in_port(1))

        tf_proposal_reshape_4d = create_op_node_with_second_input(graph, Reshape,
                                                                  int64_array([-1, 1, max_proposals, 5]),
                                                                  dict(name="reshape_proposal_4d"), proposal)
        mark_as_correct_data_layout(tf_proposal_reshape_4d)

        crop_op = Crop(graph, dict(axis=int64_array([3]), offset=int64_array([1]), dim=int64_array([4]),
                                   nchw_layout=True))
        # the crop_proposals node is used in the ObjectDetectionAPIDetectionOutputReplacement transformation
        crop = crop_op.create_node([tf_proposal_reshape_4d], dict(name='crop_proposals'))

        tf_proposals_crop_reshape_3d_node = create_op_node_with_second_input(graph, Reshape, int64_array([0, -1, 4]),
                                                                             dict(name="reshape_crop_3d"), crop)
        mark_input_as_in_correct_layout(tf_proposals_crop_reshape_3d_node, 0)
        return tf_proposals_crop_reshape_3d_node


"""
An important part of many object detection models is an operation DetectionOutput which decodes final detection boxes
using predicted bounding boxes shape offsets and prior boxes inputs. And finally performs non-maximum-suppression based
on decoded boxes and their confidences (scores). There is no DetectionOutput operation in TensorFlow operation set, it
is implemented as a sub-graph of primitive operations instead. There are two transformations which replace the sub-graph
implementing DetectionOutput operation in this file: ObjectDetectionAPISSDPostprocessorReplacement and 
ObjectDetectionAPIDetectionOutputReplacement. The first one is used for SSD models, the second one for Faster-RCNN,
Mask-RCNN and RFCN models. These transformations also prepare input data for the DetectionOutput operation because the
layout and shape of the data is different between the TensorFlow and the OpenVINO. The most notable difference
is that bounding boxes and deltas are calculated with YXYX order in the TensorFlow model whilst OpenVINO
operation DetectionOutput, ROIPooling and Proposal expects them and produce the output with XYXY order. Refer to the
transformation code and operations specifications for more details.
"""


class ObjectDetectionAPISSDPostprocessorReplacement(FrontReplacementFromConfigFileSubGraph):
    """
    The transformation replaces the TensorFlow sub-graph performing DetectionOutput with the DetectionOutput operation
    and adds some nodes to prepare input data in correct layout and shape.
    """
    replacement_id = 'ObjectDetectionAPISSDPostprocessorReplacement'
    run_not_recursively = True

    def run_after(self):
        return [ObjectDetectionAPIPreprocessorReplacement, ObjectDetectionAPIPreprocessor2Replacement]

    def run_before(self):
        return [ObjectDetectionAPITransformationsFinish]

    def output_edges_match(self, graph: Graph, match: SubgraphMatch, new_sub_graph: dict):
        # the DetectionOutput in OV produces single tensor, but in TF it produces two tensors, so create only one output
        # edge match
        return {match.output_node(0)[0].id: new_sub_graph['detection_output_node'].id}

    def generate_sub_graph(self, graph: Graph, match: SubgraphMatch):
        argv = graph.graph['cmd_params']
        if argv.tensorflow_object_detection_api_pipeline_config is None:
            raise Error(missing_param_error)
        pipeline_config = PipelineConfig(argv.tensorflow_object_detection_api_pipeline_config)

        has_background_class = _value_or_raise(match, pipeline_config, 'add_background_class')
        num_classes = _value_or_raise(match, pipeline_config, 'num_classes') + has_background_class

        # reshapes confidences to 4D before applying activation function and do not convert from NHWC to NCHW this node.
        # the add_activation_function_after_node function may insert the Softmax operation which is performed over the
        # last dimension which should have a specific size = num_classes. In the original model the last dimension may
        # be different, so this Reshape is absolutely necessary
        reshape_conf_before_ac = create_op_node_with_second_input(graph, Reshape, int64_array([0, 1, -1, num_classes]),
                                                                  {'name': 'do_ExpandDims_conf'})
        reshape_conf_before_ac.in_port(0).connect(match.input_nodes(1)[0][0].in_node(0).out_port(0))
        mark_as_correct_data_layout(reshape_conf_before_ac)

        # the transformation nodes are selected such a way that the confidences/scores post-processing activation
        # function is removed. This was done in order to support several versions of the model using one JSON config
        # file. Therefore, it is necessary to manually add this operation back to the graph
        activation_function = _value_or_raise(match, pipeline_config, 'postprocessing_score_converter')
        activation_conf_node = add_activation_function_after_node(graph, reshape_conf_before_ac, activation_function)

        # OV DetectionOutput operation expects flattened tensor with bounding boxes shape offsets, so reshaping it
        reshape_offsets = create_op_node_with_second_input(graph, Reshape, int64_array([0, -1]),
                                                           {'name': 'do_reshape_offsets'})

        # skip all Identity nodes and Reshape/Squeeze/Unsqueeze ops which may break the conversion because add or split
        # unnecessary dimensions
        current_node = skip_nodes_by_condition(match.input_nodes(0)[0][0].in_node(0),
                                               lambda x: x.op == 'Identity' or x.has_and_set('reinterp_shape'))
        reshape_offsets.in_port(0).connect(current_node.out_port(0))
        mark_as_correct_data_layout(reshape_offsets)

        # OV DetectionOutput operation expects flattened tensor with class confidences, so reshaping it
        reshape_conf_node = create_op_node_with_second_input(graph, Reshape, int64_array([0, -1]),
                                                             {'name': 'do_reshape_conf'}, activation_conf_node)
        mark_as_correct_data_layout(reshape_conf_node)

        need_swap_priors = False
        # the SSD model is a fully convolutional model so it can perform detection for the arbitrary input shape image
        # if the input with prior boxes is properly generated based on the input image size. There were some TensorFlow
        # models where this input was hardcoded as a constant and so the model can predict images of the specific input
        # size only. The code below inserts PriorBox or PriorBoxClustered operations which generate prior boxes and a
        # function call "_relax_reshape_nodes" to fix hardcoded output shapes specified for some Reshape operations in
        # the original model. These workarounds can be disabled by specifying parameter
        # 'disable_prior_boxes_layers_generator' in the JSON transformation configuration file or is automatically
        # disabled if necessary information about prior box generators is not known
        if not match.custom_replacement_desc.custom_attributes.get('disable_prior_boxes_layers_generator', False) and \
                (pipeline_config.get_param('ssd_anchor_generator_num_layers') is not None or
                 pipeline_config.get_param('multiscale_anchor_generator_min_level') is not None):
            # change the Reshape operations with hardcoded number of output elements of the convolution nodes to be
            # reshape-able
            _relax_reshape_nodes(graph, pipeline_config)

            # create PriorBoxClustered nodes instead of a constant value with prior boxes so the model could be reshaped
            if pipeline_config.get_param('ssd_anchor_generator_num_layers') is not None:
                priors_node = _create_prior_boxes_node(graph, pipeline_config)
            else:
                priors_node = _create_multiscale_prior_boxes_node(graph, pipeline_config)
        else:
            log.info('The anchor generator is not known. Save constant with prior-boxes to IR.')
            tf_priors_node = match.input_nodes(2)[0][0].in_node(0)
            # original prior boxes are stored as YXYX while DetectionOutput expects them to be represented as XYXY.
            # also variances should be encoded into this input. Variances are the values which are used during decoding
            # of bounding boxes from prior boxes and shape offsets. Refer to the DetectionOutput operation
            # implementation for more details
            flattened_priors = create_op_with_const_inputs(graph, Reshape, {1: int64_array([1, 1, -1])},
                                                           {'name': 'flattened_priors'}, tf_priors_node)
            mark_as_correct_data_layout(flattened_priors)

            # create tensor of shape [4] with variance values which then are tiled by the number of boxes which is
            # obtained from the 'priors_node' node
            priors_shape = Shape(graph, {'name': 'priors_shape'}).create_node([tf_priors_node])
            priors_count = node_to_get_shape_value_of_indices(priors_shape, [-2])

            # replicating the variance values for all prior-boxes
            variances = Const(graph, {'value': _variance_from_pipeline_config(pipeline_config)}).create_node()
            tiled_variances = Tile(graph, {'name': 'tiled_variances'}).create_node([variances, priors_count])
            flattened_tiled_variances = create_op_with_const_inputs(graph, Reshape, {1: int64_array([1, 1, -1])},
                                                                    {'name': 'flattened_tiled_variances'},
                                                                    tiled_variances)
            # now we can concatenate priors with a tensor with variances as it is expected by the DetectionOutput
            priors_node = Concat(graph, {'axis': 1, 'name': 'priors_with_variances'}).create_node(
                [flattened_priors, flattened_tiled_variances])

            # set a flag that priors should we swapped from YXYX to XYXY
            need_swap_priors = True

        detection_output_op = DetectionOutput(graph, match.custom_replacement_desc.custom_attributes)
        # during the bounding boxes detection the intermediate boxes are clipped to be in range [0, 1]. Different
        # versions of the TF OD API SSD models have this clipping at different stages. Special attributes
        # "clip_before_nms" and "clip_after_nms" were introduced to the operation DetectionOutput to handle these cases.
        # These attributes are specified in the JSON transformation configuration file
        detection_output_node = detection_output_op.create_node(
            [reshape_offsets, reshape_conf_node, priors_node],
            dict(name=detection_output_op.attrs['type'],
                 background_label_id=0 if has_background_class else -1,
                 variances_encoded_in_target=False,
                 confidence_threshold=_value_or_raise(match, pipeline_config, 'postprocessing_score_threshold'),
                 top_k=_value_or_raise(match, pipeline_config, 'postprocessing_max_detections_per_class'),
                 keep_top_k=_value_or_raise(match, pipeline_config, 'postprocessing_max_total_detections'),
                 nms_threshold=_value_or_raise(match, pipeline_config, 'postprocessing_iou_threshold')))

        # the TensorFlow model keeps the bounding boxes shape offsets as YXYX, while OV DetectionOutput expects them to
        # be specified as XYXY. The solution is to update last convolutions weights and biases to produce XY->YX swapped
        # bounding boxes offsets
        conv_nodes = backward_bfs_for_operation(detection_output_node.in_node(0), ['Conv2D'], ['ShapeOf'])
        swap_weights_xy(graph, conv_nodes)

        # also need to swap priors from YXYX to XYXY if this input was used from the original model. If the input was
        # not with PriorBox or PriorBoxClustered operations above then the layout will be XYXY
        if need_swap_priors:
            insert_weights_swap_xy_sub_graph(graph, detection_output_node.in_port(2).get_connection())

        # need to mark some Squeeze, Reshape and Concat operations to not change the layout
        mark_squeeze_reshape_concat_before_detection_output(conv_nodes)

        # As outputs are replaced with a postprocessing node, outgoing tensor names are no longer correspond to the
        # original tensors and should be removed from output->Result edges
        clear_tensor_names_info([match.output_node(out)[0] for out in range(match.outputs_count())])

        # return dictionary with mapping of nodes that is used in the `output_edges_match` function to finish sub-graph
        # replacement by re-connecting output from the original matched output node to the DetectionOutput node
        return {'detection_output_node': detection_output_node}


class ObjectDetectionAPIOutputReplacement(FrontReplacementFromConfigFileGeneral):
    """
    This replacer is used to cut-off the network by specified nodes for models generated with Object Detection API.
    The custom attribute for the replacer contains one value for key "outputs". This string is a comma separated list
    of outputs alternatives. Each output alternative is a '|' separated list of node name which could be outputs. The
    first node from each alternative group that exits in the graph is chosen. Others are ignored.
    For example, if the "outputs" is equal to the following string:

        "Reshape_16,SecondStageBoxPredictor_1/Conv_3/BiasAdd|SecondStageBoxPredictor_1/Conv_1/BiasAdd"

    then the "Reshape_16" will be an output if it exists in the graph. The second output will be
    SecondStageBoxPredictor_1/Conv_3/BiasAdd if it exist in the graph, if not then
    SecondStageBoxPredictor_1/Conv_1/BiasAdd will be output if it exists in the graph.
    """
    replacement_id = 'ObjectDetectionAPIOutputReplacement'
    run_not_recursively = True

    def run_after(self):
        return [ObjectDetectionAPITransformationsStart]

    def run_before(self):
        return [ObjectDetectionAPIPreprocessorReplacement, ObjectDetectionAPIPreprocessor2Replacement]

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
    """
    RFCN models contain a unique block ("SecondStageBoxPredictor") performing bounding boxes predictions which is
    called Position Sensitive ROI Pooling (PSROIPooling). The combination of "CropAndResize operations located in the
    "while" loop forms a single PSROIPooling operation with bilinear interpolation. The transformation matches two
    "while" loops with PSROIPooling layers applied to the tensors with box coordinates and classes predictions. The
    sub-graph being replaced also contains a Reduce operation performing mean calculation over the spatial dimensions,
    so the transformation adds this operation as well.
    """
    replacement_id = 'ObjectDetectionAPIPSROIPoolingReplacement'
    run_not_recursively = True

    def run_after(self):
        return [ObjectDetectionAPIProposalReplacement]

    def run_before(self):
        return [ObjectDetectionAPITransformationsFinish]

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

        proposal_nodes = graph.get_op_nodes(name='reshape_swap_proposals_2d')
        if len(proposal_nodes) != 1:
            raise Error("Found the following nodes '{}' with name 'reshape_swap_proposals_2d' but there should be "
                        "exactly 1. Looks like ObjectDetectionAPIProposalReplacement transformation didn't work."
                        "".format(proposal_nodes))
        reshape_swap_proposals_node = proposal_nodes[0]

        psroipooling_node = PSROIPoolingOp(graph, {'name': input_node.soft_get('name') + '/PSROIPooling',
                                                   'output_dim': psroipooling_output_dim,
                                                   'group_size': crop_width // num_spatial_bins_width,
                                                   'spatial_bins_x': num_spatial_bins_width,
                                                   'spatial_bins_y': num_spatial_bins_height,
                                                   'mode': 'bilinear',
                                                   'spatial_scale': 1,
                                                   }).create_node([input_node, reshape_swap_proposals_node])

        # add Reduce operation which is a part of the graph being removed
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
    run_not_recursively = True

    def run_after(self):
        return [ObjectDetectionAPITransformationsStart]

    def run_before(self):
        return [ObjectDetectionAPIPreprocessorReplacement, ObjectDetectionAPIPreprocessor2Replacement]

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
            node.value = mo_array(pipeline_config.get_param(pipeline_config_name))
            node.value = node.value.reshape(node.shape)
