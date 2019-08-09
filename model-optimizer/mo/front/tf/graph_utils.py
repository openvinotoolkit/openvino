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

import collections
import logging as log

import numpy as np

from extensions.back.InsertLayoutPropagationTransposes import mark_input_as_in_correct_layout, \
    mark_output_as_in_correct_layout
from extensions.ops.activation_ops import Sigmoid
from mo.front.common.partial_infer.utils import int64_array
from mo.front.extractor import update_attrs
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
            if cur_node.op == 'Reshape' and len(cur_node.out_node().shape) == 4:
                log.debug("Found reshape op with 4D output {}".format(cur_node.id))
                if cur_node.in_node(1).has_valid('value') and cur_node.in_node(1).value is not None:
                    new_shape = cur_node.in_node(1).value
                    assert new_shape[2] == 1
                    new_shape = np.delete(new_shape, 2)
                    cur_node.in_node(1).value = new_shape
                    cur_node.in_node(1).shape = np.array(new_shape.shape, dtype=np.int64)
                    # run infer function once again
                    cur_node.infer(cur_node)
                else:
                    log.warning("The reshape size is not defined!")
            if cur_node.type == 'Concat' and len(cur_node.out_node().shape) == 4:
                log.debug("Found Concat op with 4D output {}".format(cur_node.id))
                cur_node.axis = 1
                # run infer function once again
                cur_node.infer(cur_node)
                if cur_node.out_port(0).get_destination().node.op == 'Squeeze':
                    # remove Squeeze node after the Concat
                    squeeze_consumer = cur_node.out_port(0).get_destination().node.out_port(0).get_destination()
                    cur_node.out_port(0).get_connection().set_destination(squeeze_consumer)

        out_node_size = len(cur_node.out_nodes())
        for ind in range(out_node_size):
            node = cur_node.out_node(ind)
            q.append(node)


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
    """
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
