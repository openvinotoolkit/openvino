# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import collections
import logging as log
from typing import Dict

import numpy as np

from extensions.middle.InsertLayoutPropagationTransposes import mark_input_as_in_correct_layout, \
    mark_output_as_in_correct_layout
from extensions.ops.activation_ops import Sigmoid
from extensions.ops.elementwise import Add, Less, Mul
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node, Graph
from mo.ops.concat import Concat
from mo.ops.const import Const
from mo.ops.convolution import Convolution
from mo.ops.crop import Crop
from mo.ops.reshape import Reshape
from mo.ops.softmax import Softmax
from mo.utils.error import Error


def create_op_node_with_second_input(graph: Graph, op: callable, second_input_value: np.array, op_attrs=None,
                                     input_node=None):
    operation = op(graph, op_attrs)
    node = operation.create_node()
    if input_node is not None:
        input_node.out_port(0).connect(node.in_port(0))
    second_input_node = Const(graph, {'name': node.name + '/value', 'value': second_input_value}).create_node()
    second_input_node.out_port(0).connect(node.in_port(1))
    if graph.stage != 'front':
        second_input_node.infer(second_input_node)
    return node


def create_op_with_const_inputs(graph: Graph, op: callable, port_value_dict: Dict[int, np.array],
                                op_attrs=None, input_node=None):
    operation = op(graph, op_attrs)
    node = operation.create_node()
    if input_node is not None:
        input_node.out_port(0).connect(node.in_port(0))

    for idx, value in port_value_dict.items():
        node.add_input_port(idx, skip_if_exist=True)
        value_input_node = Const(graph, {'name': node.name + '_input_port_' + str(idx) + '/value',
                                         'value': value}).create_node()
        value_input_node.out_port(0).connect(node.in_port(idx))
        if graph.stage != 'front':
            value_input_node.infer(value_input_node)
    return node


def mark_squeeze_reshape_concat_before_detection_output(start_nodes: list):
    """
    The function looks for Reshape, Concat and Squeeze ops after the 'start_nodes' with 4D output and marks them with
    proper attributes to infer them in original NHWC layout. This is a case of the TensorFlow Object Detection API
    models for the SSD heads output which produces 4D tensor with bounding box deltas.
    :param start_nodes: list of nodes to start search from.
    :return: None
    """
    q = collections.deque()
    q.extend(start_nodes)
    while len(q) != 0:
        cur_node = q.popleft()
        if cur_node.has_valid('type'):
            if cur_node.soft_get('type') == 'DetectionOutput':  # do not go beyond the DetectionOutput node
                continue
            # the input to Reshape comes from Convolution so it will be converted from NCHW to NHWC layout in the
            # InsertLayoutPropagationTransposes transformation. But the output should be kept in the original layout
            if cur_node.soft_get('type') == 'Reshape' and len(cur_node.out_port(0).data.get_shape()) == 4:
                mark_output_as_in_correct_layout(cur_node, 0)

            # Concat should be inferred in the original layout so the input with concatenation axis should not be
            # updated from NHWC to NCHW layout
            if cur_node.soft_get('type') == 'Concat' and len(cur_node.out_port(0).data.get_shape()) == 4:
                cur_node.in_port(1).__setattr__('input_permutation', None)
                cur_node['nchw_layout'] = True
                cur_node.out_node(0)['nchw_layout'] = True

            # Squeeze should be inferred in the original layout so the input with squeeze axis should not be updated
            # from NHWC to NCHW layout. The input is marked as in correct layout to prevent from inserting Transpose
            # from NHWC to NCHW.
            if cur_node.soft_get('type') == 'Squeeze' and len(cur_node.in_port(0).data.get_shape()) == 4:
                cur_node.in_port(1).__setattr__('input_permutation', None)
                mark_input_as_in_correct_layout(cur_node, 0)

        [q.append(port.node) for port in cur_node.out_port(0).get_destinations()]


def add_convolution_to_swap_xy_coordinates(graph: Graph, input_node: Node, coordinates_size: int):
    """
    The function add convolution node after the node 'input_node' to swap xy coordinates of the boxes produced
    by the node 'input_node'. It is expected that box coordinates are located in the fastest changing dimension of the
    'input_node' output, i.e. the input tensor could be reshaped to [num_boxes, 4] or [num_boxes, 5]. If the size is 5,
    then the 0-th element for each of num_boxes blocks is not changed and element 1 is swapped with element 2, element 3
    is swapped with element 4. This is the case when boxes coordinates are produced by the layer "Proposal". The exact
    amount of elements in each block is equal to the 'coordinates_size' parameter.
    :param graph: graph to operate on.
    :param input_node: node producing boxes coordinates.
    :param coordinates_size: integer value equal to 4 or 5.
    :return convolution node that swaps coordinates.
    """
    # swap of input tensor with 4 or 5 numbers describing boxes are supported
    assert (coordinates_size in [4, 5])

    input_reshape_4d_node = create_op_node_with_second_input(graph, Reshape, int64_array([-1, 1, 1, coordinates_size]),
                                                             dict(name=input_node.name + '/reshape_4d'), input_node)
    mark_input_as_in_correct_layout(input_reshape_4d_node, 0)
    # do not mark second input because the reshape works in initial model layout and needs to be transformed to NCHW
    mark_output_as_in_correct_layout(input_reshape_4d_node, 0)

    if coordinates_size == 5:
        # zero indexed element is not box coordinate ("batch id" in case of Proposal)
        conv_filter_data = np.array(np.array([[[[1, 0, 0, 0, 0],
                                                [0, 0, 1, 0, 0],
                                                [0, 1, 0, 0, 0],
                                                [0, 0, 0, 0, 1],
                                                [0, 0, 0, 1, 0]]]],
                                             dtype=np.float32))
    else:
        conv_filter_data = np.array(np.array([[[[0, 1, 0, 0],
                                                [1, 0, 0, 0],
                                                [0, 0, 0, 1],
                                                [0, 0, 1, 0]]]],
                                             dtype=np.float32))

    conv_filter_data = np.transpose(conv_filter_data, [2, 3, 0, 1])

    conv_filter_const_op = Const(graph, dict(value=conv_filter_data))
    conv_filter_const_node = conv_filter_const_op.create_node([], dict(name=input_node.name + '/weights'))

    conv_op = Convolution(graph, {
        'bias_addable': True,
        'channel_dims': np.array([3]),
        'batch_dims': np.array([0]),
        'input_feature_channel': 0,
        'output_feature_channel': 1,
        'group': 1,
        'layout': 'NHWC',
    })
    return conv_op.create_node([input_reshape_4d_node, conv_filter_const_node], dict(name=input_node.name + "/conv"))


def add_fake_background_loc(graph: Graph, input_node: Node):
    r"""
    DetectionOutput layer expects that box coordinates contains coordinates of boxes for the "background" class also,
    but in the TensorFlow\* Object Detection API the tensor contains information about real object classes only.
    The function copies a slice of the output data of the node 'input_node' and then concats it to the beginning of the
    data. The data in this slice is not used by the Detection Output layer so the actual values are not important. This
    approach allows the model to be reshape-able and does not introduce many layers.
    "background" class box coordinates.
    :param graph: graph to operate on.
    :param input_node: node producing the boxes coordinates.
    :return convolution node that adds slice of data for the "background" class.
    """
    crop_op = Crop(graph, dict(axis=np.array([1]), offset=np.array([0]), dim=np.array([1]), nchw_layout=True))
    crop_node = crop_op.create_node([input_node], dict(name='crop_locs'))

    concat_op = Concat(graph, dict(axis=1, in_ports_count=2, nchw_layout=True))
    return concat_op.create_node([crop_node, input_node], dict(name=input_node.id + '/locs_with_fake_background'))


def add_activation_function_after_node(graph: Graph, node: Node, activation_function: str):
    """
    The function adds node with activation function defined by string 'activation_function' which gets input from the
    node 'node'.
    :param graph: graph to operate on.
    :param node: node to add activation after.
    :param activation_function: string defining the activation function. These values are read from TensorFlow* object
    detection API pipeline configuration file
    :return: activation function node.
    """
    if activation_function == 'SOFTMAX':
        # softmax to be applied to the confidence
        softmax_conf_op = Softmax(graph, dict(axis=-1, nchw_layout=True))
        activation_node = softmax_conf_op.create_node([node], dict(name=node.name + '/softmax'))
    elif activation_function == 'SIGMOID':
        # sigmoid activation function to be applied to the confidence
        sigmoid_conf_op = Sigmoid(graph, dict(nchw_layout=True))
        activation_node = sigmoid_conf_op.create_node([node], dict(name=node.name + '/sigmoid'))
    elif activation_function == 'IDENTITY':
        # in case of Identity do nothing and just use result from the input node
        activation_node = node
    else:
        raise Error('Unknown post-processing activation function "{}".'.format(activation_function))
    return activation_node


def add_constant_to_negative_values(node: Node, port_idx: int, added_value: np.array):
    """
    This function adds the given values to negative elements of value from the given input port.
    :param node: node with corrected values in the input port port_idx
    :param port_idx: input port index for negative values
    :param added_value: the value to add
    :return: None
    """
    negative_values_source = node.in_port(port_idx).get_source()
    negative_values_node = node.in_port(port_idx).get_source().node
    negative_values_node_name = negative_values_node.soft_get('name', negative_values_node.id)

    graph = node.graph

    less_node = create_op_with_const_inputs(graph, Less,
                                            {1: np.array(0, dtype=added_value.dtype)},
                                            {'name': negative_values_node_name + '/Less'})
    mul_node = create_op_with_const_inputs(graph, Mul, {1: added_value}, {'name': negative_values_node_name + '/Mul'})

    node.in_port(port_idx).get_connection().set_destination(less_node.in_port(0))
    less_node.out_port(0).connect(mul_node.in_port(0))

    add_node = Add(graph, {}).create_node()
    mul_node.out_port(0).connect(add_node.in_port(1))
    negative_values_source.connect(add_node.in_port(0))
    add_node.out_port(0).connect(node.in_port(port_idx))
