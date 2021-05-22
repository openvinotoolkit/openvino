# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
from math import sqrt

import numpy as np

from extensions.front.Pack import Pack
from extensions.front.TransposeOrderNormalizer import TransposeOrderNormalizer
from extensions.front.split_normalizer import SqueezeAxis
from extensions.front.tf.CropAndResizeReplacement import CropAndResizeReplacement
from extensions.front.tf.FakeQuantWithMinMaxVars import FakeQuantWithMinMaxVarsToQuantize
from extensions.front.tf.KerasRNNTransformation import KerasRNNInputSlicing, KerasRNNOutputConcatenation
from extensions.front.tf.TFSliceToSlice import TFSliceToSliceReplacer
from extensions.front.tf.pad_tf_to_pad import PadTFToPad
from extensions.middle.InsertLayoutPropagationTransposes import mark_as_correct_data_layout, \
    mark_input_as_in_correct_layout, mark_output_as_in_correct_layout
from extensions.ops.DetectionOutput import DetectionOutput
from extensions.ops.ReduceOps import ReduceMean
from extensions.ops.activation_ops import Sigmoid
from extensions.ops.elementwise import Mul, Sub, Add, Div
from extensions.ops.gather import Gather
from extensions.ops.parameter import Parameter
from extensions.ops.priorbox_clustered import PriorBoxClusteredOp
from extensions.ops.proposal import ProposalOp
from extensions.ops.psroipooling import PSROIPoolingOp
from extensions.ops.transpose import Transpose
from mo.front.common.layout import get_batch_dim, get_height_dim, get_width_dim
from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementPattern
from mo.front.extractor import output_user_data_repack, add_output_ops
from mo.front.subgraph_matcher import SubgraphMatch
from mo.front.tf.graph_utils import add_activation_function_after_node, add_convolution_to_swap_xy_coordinates, \
    mark_squeeze_reshape_concat_before_detection_output, add_fake_background_loc, create_op_node_with_second_input, \
    create_op_with_const_inputs
from mo.front.tf.replacement import FrontReplacementFromConfigFileSubGraph, FrontReplacementFromConfigFileGeneral
from mo.graph.graph import Graph, Node
from mo.ops.concat import Concat
from mo.ops.const import Const
from mo.ops.crop import Crop
from mo.ops.op import PermuteAttrs
from mo.ops.reshape import Reshape
from mo.ops.result import Result
from mo.ops.roipooling import ROIPooling
from mo.ops.shape import Shape
from mo.ops.softmax import Softmax
from mo.utils.error import Error
from mo.utils.graph import backward_bfs_for_operation, bfs_search, clear_tensor_names_info, sub_graph_between_nodes
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
        prior_box_op = PriorBoxClusteredOp(graph, {'width': np.array(widths), 'height': np.array(heights),
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


def skip_nodes_by_condition(current_node: Node, condition: callable):
    while condition(current_node):
        current_node = current_node.in_node()
    return current_node


def calculate_shape_keeping_aspect_ratio(height: int, width: int, min_size: int, max_size: int,
                                         pad_to_max_dimension: bool = False):
    """
    The function scales spatial sizes of the image keeping aspect ratio to satisfy provided requirements.
    The behavior of this function is equivalent to the output shape calculation of the Preprocessor block of TensorFlow
    Object Detection API models with keep aspect ratio resizer.
    :param height: input height.
    :param width: input width.
    :param min_size: size limit.
    :param max_size: size limit.
    :param pad_to_max_dimension: scale the input image size to the maximum value specified
    :return: the tuple with scaled image height, width.
    """
    if pad_to_max_dimension:
        return max_size, max_size
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
        pad_to_max_dimension = pipeline_config.get_param('pad_to_max_dimension')
        print('[ WARNING ] Model Optimizer removes pre-processing block of the model which resizes image keeping '
              'aspect ratio. The Inference Engine does not support dynamic image size so the Intermediate '
              'Representation file is generated with the input image size of a fixed size.')
        if user_defined_height and user_defined_width:
            scaled_height, scaled_width = calculate_shape_keeping_aspect_ratio(user_defined_height,
                                                                               user_defined_width,
                                                                               resizer_min_dimension,
                                                                               resizer_max_dimension,
                                                                               pad_to_max_dimension)
            if scaled_height != user_defined_height or scaled_width != user_defined_width:
                log.error('The model resizes the input image keeping aspect ratio with min dimension {}, max '
                          'dimension {}. The provided input height {}, width {} is transformed to height {}, width '
                          '{}.'.format(resizer_min_dimension, resizer_max_dimension, user_defined_height,
                                       user_defined_width, scaled_height, scaled_width), extra={'is_warning': True})
            height = scaled_height
            width = scaled_width
        else:
            if pad_to_max_dimension:
                height = width = resizer_max_dimension
            else:
                height = width = resizer_min_dimension
            print('Specify the "--input_shape" command line parameter to override the default shape which is equal to '
                  '({}, {}).'.format(height, width))

    if height is None or width is None:
        raise Error('Failed to determine the placeholder shape.')
    return height, width


def update_parameter_shape(graph: Graph, match: [SubgraphMatch, None]):
    """
    Updates the shape of the model Parameter node based on the user provided input shape or values provided in the
    pipeline.config configuration file used for model training.
    :param graph: model graph
    :param match: Match object with information abouot matched sub-graph
    :return: tupe with input node names and Parameter Node
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
    if argv.batch is None and parameter_node.shape[batch_dim] == -1:
        parameter_node.shape[batch_dim] = 1
    height, width = calculate_placeholder_spatial_shape(graph, match, pipeline_config)
    parameter_node.shape[get_height_dim(layout, 4)] = height
    parameter_node.shape[get_width_dim(layout, 4)] = width

    # save the pre-processed image spatial sizes to be used in the other replacers
    graph.graph['preprocessed_image_height'] = parameter_node.shape[get_height_dim(layout, 4)]
    graph.graph['preprocessed_image_width'] = parameter_node.shape[get_width_dim(layout, 4)]
    return initial_input_node_name, parameter_node


class ObjectDetectionAPITransformationsStart(FrontReplacementPattern):
    """
    This is a anchor transformation which is used to distinguish TF OD API models related transformations.
    """
    enabled = True

    def run_after(self):
        return [CropAndResizeReplacement, FakeQuantWithMinMaxVarsToQuantize]

    def find_and_replace_pattern(self, graph: Graph):
        pass


class ObjectDetectionAPITransformationsFinish(FrontReplacementPattern):
    """
    This is a anchor transformation which is used to distinguish TF OD API models related transformations.
    """
    enabled = True
    # cleanup the graph after applying of TF OD API transformations to remove a lot of unconnected nodes to avoid issues
    # with shape inference
    force_clean_up = True

    def run_before(self):
        # PadTFToPad inserts Transpose ops for Pad ops inside the sub-graph corresponding to DetectionOutput.
        # But the inputs corresponding to padding values is re-used as inputs for newly created Pad node. This input
        # is removed during removing nodes from the DO sub-graph so the first input to Transpose is missing which
        # results in TransposeOrderNormalizer transformation failure.
        return [Pack, TransposeOrderNormalizer, PadTFToPad, SqueezeAxis, TFSliceToSliceReplacer,
                KerasRNNOutputConcatenation, KerasRNNInputSlicing]

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
    return preprocessing_nodes, trailing


class ObjectDetectionAPIPreprocessorReplacement(FrontReplacementFromConfigFileSubGraph):
    """
    The class replaces the "Preprocessor" block resizing input image and applying mean/scale values. Only nodes related
    to applying mean/scaling values are kept.
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
        if sub_node.soft_get('op') != 'Sub':
            raise Error('The output op of the Preprocessor sub-graph is not of type "Sub". Looks like the topology is '
                        'not created with TensorFlow Object Detection API.')

        mul_node = None
        if sub_node.in_port(0).get_source().node.soft_get('op') == 'Mul':
            log.info('There is image scaling node in the Preprocessor block.')
            mul_node = sub_node.in_port(0).get_source().node

        initial_input_node_name, placeholder_node = update_parameter_shape(graph, match)

        to_float_node = placeholder_node.out_port(0).get_destination().node
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

        print('The Preprocessor block has been removed. Only nodes performing mean value subtraction and scaling (if'
              ' applicable) are kept.')
        return {}


class ObjectDetectionAPIPreprocessor2Replacement(FrontReplacementFromConfigFileGeneral):
    """
    The class replaces the "Preprocessor" block resizing input image and applying mean/scale values. Only nodes related
    to applying mean/scaling values are kept. The transformation is used for TensorFlow 2.X models.

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
    or keeps the existing one if they are in the main graph originally.
    """
    replacement_id = 'ObjectDetectionAPIPreprocessor2Replacement'
    run_not_recursively = True

    def run_before(self):
        return [ObjectDetectionAPITransformationsFinish]

    def run_after(self):
        return [ObjectDetectionAPITransformationsStart]

    def transform_graph(self, graph: Graph, replacement_descriptions: dict):
        update_parameter_shape(graph, None)

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
        loop_nodes_ids = [node_id for node_id in sub_graph_node_ids if graph.node[node_id].get('op') == 'Loop']
        if len(loop_nodes_ids):
            assert len(loop_nodes_ids) == 1, 'There should be exactly one Loop node in the pre-processor block.'
            pre_processing_in_loop = True
            loop_node = Node(graph, loop_nodes_ids[0])
            body_graph = loop_node.body
            # we stick to the nodes with ids 'map/while/Preprocessor/unstack' and 'map/while/Preprocessor/stack' as they
            # "wrap" nodes performing image resize. The scale/mean values nodes are located strictly before or after
            # them
            pre_processing_ops, trailing = get_preprocessing_ops(body_graph,
                                                                 'map/while/Preprocessor/unstack',
                                                                 'map/while/Preprocessor/stack')
        else:
            pre_processing_ops, trailing = get_preprocessing_ops(graph, start_node.id, end_node.id)

        if len(pre_processing_ops):
            # if the pre-processing is applied before the resize then reverse them to be in the topological order
            if not trailing:
                pre_processing_ops = list(reversed(pre_processing_ops))

            if pre_processing_in_loop:  # case 4 and 5
                # build a sub-graph containing a sequence of pre_processing_ops if they came from the Loop
                new_preprocessing_ops = []
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
                    new_preprocessing_ops[0].in_port(new_preprocessing_ops[0].is_in_port_connected(0)))
            else:
                if trailing:  # case 2
                    # change output of the end_node to be produced with the start node producer
                    source_port = start_node.in_port(0).get_source()
                    source_port.disconnect()
                    end_node.out_port(0).get_connection().set_source(source_port)
                else:  # case 1
                    # change output of the end_node to be produced with the last preprocessing op
                    end_node.out_port(0).get_connection().set_source(pre_processing_ops[-1][0].out_port(0))
                    start_node.in_port(0).disconnect()
        else:  # simply remove the nodes in between start_node and end_node (including them). Case 3 and 6
            end_node.out_port(0).get_connection().set_source(start_node.in_port(0).get_source())

        print('The Preprocessor block has been removed. Only nodes performing mean value subtraction and scaling (if'
              ' applicable) are kept.')


class ObjectDetectionAPIDetectionOutputReplacement(FrontReplacementFromConfigFileSubGraph):
    """
    Replaces the sub-graph that is equal to the DetectionOutput layer from Inference Engine. This replacer is used for
    Faster R-CNN, R-FCN and Mask R-CNN topologies conversion.
    The replacer uses a value of the custom attribute 'coordinates_swap_method' from the sub-graph replacement
    configuration file to choose how to swap box coordinates of the 0-th input of the generated DetectionOutput layer.
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
        # the DetectionOutput in IE produces single tensor, but in TF it produces four tensors, so we need to create
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

        share_box_across_classes = _value_or_raise(match, pipeline_config, 'share_box_across_classes')
        background_label_id = int(custom_attributes.get('background_label_id', 0))
        if share_box_across_classes:
            reshape_loc_node = create_op_node_with_second_input(graph, Reshape, int64_array([-1, 1, 1, 4]),
                                                                dict(name='reshape_loc'), current_node)
        else:
            reshape_loc_node = create_op_node_with_second_input(graph, Reshape, int64_array([-1, num_classes, 1, 4]),
                                                                dict(name='reshape_loc'), current_node)
        mark_as_correct_data_layout(reshape_loc_node)

        # constant node with variances
        variances_const_op = Const(graph, dict(value=_variance_from_pipeline_config(pipeline_config)))
        variances_const_node = variances_const_op.create_node([])

        if share_box_across_classes:
            loc_node = reshape_loc_node
        else:
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

        if share_box_across_classes:
            reshape_loc_do_dims = Const(graph, {'value': int64_array([-1, max_proposals * 4]),
                                                'name': reshape_loc_do_node.name + '/Dim'}).create_node()
        else:
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
        for key in ('clip_before_nms', 'clip_after_nms'):
            if key in match.custom_replacement_desc.custom_attributes:
                detection_output_op.attrs[key] = int(match.custom_replacement_desc.custom_attributes[key])

        detection_output_node = detection_output_op.create_node(
            [reshape_loc_do_node, reshape_conf_node, reshape_priors_node],
            dict(name=detection_output_op.attrs['type'],
                 share_location=int(share_box_across_classes),
                 variance_encoded_in_target=1,
                 background_label_id=background_label_id,
                 code_type='caffe.PriorBoxParameter.CENTER_SIZE', pad_mode='caffe.ResizeParameter.CONSTANT',
                 resize_mode='caffe.ResizeParameter.WARP',
                 num_classes=num_classes,
                 confidence_threshold=_value_or_raise(match, pipeline_config, 'postprocessing_score_threshold'),
                 top_k=_value_or_raise(match, pipeline_config, 'postprocessing_max_detections_per_class'),
                 keep_top_k=_value_or_raise(match, pipeline_config, 'postprocessing_max_total_detections'),
                 nms_threshold=_value_or_raise(match, pipeline_config, 'postprocessing_iou_threshold')))
        # sets specific name to the node so we can find it in other replacers
        detection_output_node.name = 'detection_output'

        if coordinates_swap_method == 'swap_weights':
            swap_weights_xy(graph, backward_bfs_for_operation(detection_output_node.in_node(0), ['MatMul', 'Conv2D']))

        # when the use_matmul_crop_and_resize = True then the prior boxes were not swapped and we need to swap them from
        # YXYX to XYXY before passing to the DetectionOutput operation
        if pipeline_config.get_param('use_matmul_crop_and_resize'):
            insert_weights_swap_xy_sub_graph(graph, detection_output_node.in_port(2).get_connection())
        output_op = Result(graph, dict(name='do_OutputOp'))
        output_op.create_node([detection_output_node])

        print('The graph output nodes have been replaced with a single layer of type "DetectionOutput". Refer to the '
              'operation set specification documentation for more information about the operation.')

        return {'detection_output_node': detection_output_node}


class ObjectDetectionAPIMaskRCNNROIPoolingSecondReplacement(FrontReplacementFromConfigFileSubGraph):
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
        roi_pool_size = _value_or_raise(match, pipeline_config, 'initial_crop_size')

        detection_output_nodes_ids = [node_id for node_id, attrs in graph.nodes(data=True)
                                      if 'name' in attrs and attrs['name'] == 'detection_output']
        if len(detection_output_nodes_ids) != 1:
            raise Error("Found the following nodes '{}' with 'detection_output' but there should be exactly 1.".
                        format(detection_output_nodes_ids))
        detection_output_node = Node(graph, detection_output_nodes_ids[0])
        output_nodes = [port.node for port in detection_output_node.out_port(0).get_destinations() if port.node.soft_get('type') == 'Result']
        if len(output_nodes) == 1:
            graph.remove_node(output_nodes[0].id)

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
    run_not_recursively = True

    def run_before(self):
        return [ObjectDetectionAPITransformationsFinish]

    def run_after(self):
        return [ObjectDetectionAPIMaskRCNNROIPoolingSecondReplacement]

    def transform_graph(self, graph: Graph, replacement_descriptions):
        masks_node_prefix_name = replacement_descriptions.get('masks_node_prefix_name', 'SecondStageBoxPredictor')
        op_outputs = graph.get_op_nodes(op='Result')
        for op_output in op_outputs:
            last_node = op_output.in_port(0).get_source().node
            if last_node.name.startswith(masks_node_prefix_name):
                sigmoid_node = Sigmoid(graph, dict(name='masks')).create_node()
                op_output.in_port(0).get_connection().insert_node(sigmoid_node)

        print('The predicted masks are produced by the "masks" layer for each bounding box generated with a '
              '"detection_output" operation.\n Refer to operation specification in the documentation for information '
              'about the DetectionOutput operation output data interpretation.\n'
              'The model can be inferred using the dedicated demo "mask_rcnn_demo" from the OpenVINO Open Model Zoo.')


class ObjectDetectionAPIProposalReplacement(FrontReplacementFromConfigFileSubGraph):
    """
    This class replaces sub-graph of operations with Proposal layer and additional layers transforming
    tensors from layout of TensorFlow to layout required by Inference Engine.
    Refer to comments inside the function for more information about performed actions.
    """
    replacement_id = 'ObjectDetectionAPIProposalReplacement'
    run_not_recursively = True

    def run_after(self):
        return [ObjectDetectionAPIPreprocessorReplacement, ObjectDetectionAPIPreprocessor2Replacement]

    def run_before(self):
        return [ObjectDetectionAPITransformationsFinish]

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

        # models with use_matmul_crop_and_resize = True should not swap order of elements (YX to XY) after the Proposal
        swap_proposals = not match.custom_replacement_desc.custom_attributes.get('do_not_swap_proposals', False) and \
                         not pipeline_config.get_param('use_matmul_crop_and_resize')

        if swap_proposals:
            proposal_node = add_convolution_to_swap_xy_coordinates(graph, proposal_node, 5)

        proposal_reshape_2d_node = create_op_node_with_second_input(graph, Reshape, int64_array([-1, 5]),
                                                                    dict(name="reshape_swap_proposals_2d"),
                                                                    proposal_node)
        mark_input_as_in_correct_layout(proposal_reshape_2d_node, 0)

        crop_and_resize_nodes_ids = [node_id for node_id in bfs_search(graph, [match.single_input_node(0)[0].id]) if
                                     graph.node[node_id]['op'] == 'CropAndResize']
        if len(crop_and_resize_nodes_ids) != 0 and swap_proposals:
            # feed the CropAndResize node with a correct boxes information produced with the Proposal layer
            # find the first CropAndResize node in the BFS order. This is needed in the case when we already swapped
            # box coordinates data after the Proposal node
            crop_and_resize_node = Node(graph, crop_and_resize_nodes_ids[0])
            # set a marker that an input with box coordinates has been pre-processed so the CropAndResizeReplacement
            # transform doesn't try to merge the second and the third inputs
            crop_and_resize_node['inputs_preprocessed'] = True
            crop_and_resize_node.in_port(1).disconnect()
            proposal_reshape_2d_node.out_port(0).connect(crop_and_resize_node.in_port(1))

        tf_proposal_reshape_4d_node = create_op_node_with_second_input(graph, Reshape,
                                                                       int64_array([-1, 1, max_proposals, 5]),
                                                                       dict(name="reshape_proposal_4d"),
                                                                       proposal_node)

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
    run_not_recursively = True

    def run_after(self):
        return [ObjectDetectionAPIPreprocessorReplacement, ObjectDetectionAPIPreprocessor2Replacement]

    def run_before(self):
        return [ObjectDetectionAPITransformationsFinish]

    def output_edges_match(self, graph: Graph, match: SubgraphMatch, new_sub_graph: dict):
        # the DetectionOutput in IE produces single tensor, but in TF it produces two tensors, so create only one output
        # edge match
        return {match.output_node(0)[0].id: new_sub_graph['detection_output_node'].id}

    def generate_sub_graph(self, graph: Graph, match: SubgraphMatch):
        argv = graph.graph['cmd_params']
        if argv.tensorflow_object_detection_api_pipeline_config is None:
            raise Error(missing_param_error)
        pipeline_config = PipelineConfig(argv.tensorflow_object_detection_api_pipeline_config)

        has_background_class = _value_or_raise(match, pipeline_config, 'add_background_class')
        num_classes = _value_or_raise(match, pipeline_config, 'num_classes') + has_background_class

        # reshapes confidences to 4D before applying activation function and do not convert from NHWC to NCHW this node
        expand_dims_node = create_op_node_with_second_input(graph, Reshape, int64_array([0, 1, -1, num_classes]),
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
            # reshape-able
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
        for key in ('clip_before_nms', 'clip_after_nms'):
            if key in match.custom_replacement_desc.custom_attributes:
                detection_output_op.attrs[key] = int(match.custom_replacement_desc.custom_attributes[key])
        detection_output_op.attrs['old_infer'] = detection_output_op.attrs['infer']
        detection_output_op.attrs['infer'] = __class__.do_infer
        detection_output_node = detection_output_op.create_node(
            [reshape_loc_node, reshape_conf_node, priors_node],
            dict(name=detection_output_op.attrs['type'],
                 confidence_threshold=_value_or_raise(match, pipeline_config, 'postprocessing_score_threshold'),
                 top_k=_value_or_raise(match, pipeline_config, 'postprocessing_max_detections_per_class'),
                 keep_top_k=_value_or_raise(match, pipeline_config, 'postprocessing_max_total_detections'),
                 nms_threshold=_value_or_raise(match, pipeline_config, 'postprocessing_iou_threshold')))

        # compared to the IE's DetectionOutput, the TF keeps the locations in YXYX, need to get back to the XYXY
        # for last convolutions that operate the locations need to swap the X and Y for output feature weights & biases
        conv_nodes = backward_bfs_for_operation(detection_output_node.in_node(0), ['Conv2D'])
        swap_weights_xy(graph, conv_nodes)

        # As outputs are replaced with a postprocessing node, outgoing tensor names are no longer
        # correspond to original tensors and should be removed from output->Result edges
        out_nodes = []
        for out in range(match.outputs_count()):
            out_nodes.append(match.output_node(out)[0])
        clear_tensor_names_info(out_nodes)

        return {'detection_output_node': detection_output_node}

    @staticmethod
    def do_infer(node: Node):
        graph = node.graph
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

            # create Const node with an updated prior boxes values. Cannot use Port/Connection API here because we are
            # in the middle of the partial inference phase and graph is in the intermediate step
            graph.remove_edge(node.in_node(2).in_node(0).id, node.in_node(2).id)
            const = Const(graph, {'name': 'prior_boxes', 'executable': True, 'value': prior_boxes}).create_node()
            graph.create_edge(const, node.in_node(2))

        node.old_infer(node)

        conv_nodes = backward_bfs_for_operation(node.in_node(0), ['Conv2D'])
        mark_squeeze_reshape_concat_before_detection_output(conv_nodes)


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
            node.value = np.array(pipeline_config.get_param(pipeline_config_name))
            node.value = node.value.reshape(node.shape)
