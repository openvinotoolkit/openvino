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

from mo.front.extractor import update_attrs
from mo.graph.graph import Node
from mo.ops.activation import Activation
from mo.ops.const import Const
from mo.ops.convolution import Convolution
from mo.ops.reshape import Reshape
from mo.ops.softmax import Softmax
from mo.utils.error import Error


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
                    cur_node['dim'] = new_shape.copy()
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


def add_convolution_to_swap_xy_coordinates(graph: nx.MultiDiGraph, input_node: Node, coordinates_size: int):
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
    input_reshape_4d_op = Reshape(input_node.graph, dict(dim=np.array([-1, 1, 1, coordinates_size])))
    input_reshape_4d_node = input_reshape_4d_op.create_node([input_node], dict(name=input_node.name + '/reshape_4d'))
    update_attrs(input_reshape_4d_node, 'shape_attrs', 'dim')

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

    conv_filter_const_op = Const(graph, dict(value=conv_filter_data))
    conv_filter_const_node = conv_filter_const_op.create_node([], dict(name=input_node.name + '/weights'))

    conv_op = Convolution(graph, {
        'bias_addable': True,
        'channel_dims': np.array([3]),
        'batch_dims': np.array([0]),
        'input_feature_channel': 2,
        'output_feature_channel': 3,
        'group': 1,
        'layout': 'NHWC',
    })
    return conv_op.create_node([input_reshape_4d_node, conv_filter_const_node], dict(name=input_node.name + "/conv"))


def add_activation_function_after_node(graph: nx.MultiDiGraph, node: Node, activation_function: str):
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
        softmax_conf_op = Softmax(graph, dict(axis=1, nchw_layout=True))
        activation_node = softmax_conf_op.create_node([node], dict(name=node.name + '/softmax'))
    elif activation_function == 'SIGMOID':
        # sigmoid activation function to be applied to the confidence
        sigmoid_conf_op = Activation(graph, dict(operation='sigmoid', nchw_layout=True))
        activation_node = sigmoid_conf_op.create_node([node], dict(name=node.name + '/sigmoid'))
    elif activation_function == 'IDENTITY':
        # in case of Identity do nothing and just use result from the input node
        activation_node = node
    else:
        raise Error('Unknown post-processing activation function "{}".'.format(activation_function))
    return activation_node
