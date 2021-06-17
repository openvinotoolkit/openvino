# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
import re
from collections import defaultdict
from copy import copy

import numpy as np

from mo.utils.error import Error

from ngraph.frontend import InputModel


def decode_name_with_port(input_model: InputModel, node_name: str):
    """
    Decode name with optional port specification w/o traversing all the nodes in the graph
    :param input_model: Input Model
    :param node_name:
    :return: decoded place in the graph
    """
    # Check exact match with one of the names in the graph first
    node = input_model.get_place_by_tensor_name(node_name)
    if node:
        return node
    # TODO: not tested for available frontends
    regexpPost = r'(.*)(:(\d+))'
    matchPost = re.search(regexpPost, node_name)
    nodePost = input_model.get_place_by_tensor_name(matchPost.group(1)) if matchPost else None
    regexpPre = r'((\d+):)(.*)'
    matchPre = re.search(regexpPre, node_name)
    nodePre = input_model.get_place_by_tensor_name(matchPre.group(3)) if matchPost else None
    if nodePost and nodePre:
        raise Error('Name collision for {}'.format(node_name))
    if nodePost:
        return node.get_output_port(int(matchPost.group(3)))
    if nodePre:
        return node.get_input_port(int(matchPre.group(1)))
    raise Error('There is no node with name {}'.format(node_name))


def fe_input_user_data_repack(input_model: InputModel, input_user_shapes: [None, list, dict, np.ndarray],
                              freeze_placeholder: dict, input_user_data_types=dict()):
    """
    Restructures user input cutting request. Splits ports out of node names.
        Transforms node names to node ids.
    :param input_model: ngraph.frontend.InputModel - current model
    :param input_user_shapes: data structure representing user input cutting request. It may be:
    # None value if user did not provide neither --input nor --input_shape keys
    # list instance which contains input layer names with or without ports if user provided
        only --input key
    # dict instance which contains input layer names with or without ports as keys and shapes as
        values if user provided both --input and --input_shape
    # np.ndarray if user provided only --input_shape key
    :param freeze_placeholder: dictionary with placeholder names as keys and freezing value as values
    :param input_user_data_types: dictionary with input nodes and its data types
    :return: restructured input shapes and freeze placeholder shapes information
    Example of input dictionary:
    _input_shapes =
    {
        'node_ID':
            [
                {'shape': None, 'in': 0},
                {'shape': None, 'in': 1},
            ],
        'node_1_ID':
            [
                {'shape': [1, 227, 227, 3], 'port': None, 'data_type': np.int32}
            ],
        'node_2_ID':
            [
                {'shape': None, 'out': 3}
            ]
    }
     Example of freeze placeholder dictionary:
    _freeze_placeholder =
    {
        'phase_train' : False
    }
    """
    _input_shapes = []
    if isinstance(input_user_shapes, list) or isinstance(input_user_shapes, dict):
        for input_name in input_user_shapes:
            node = decode_name_with_port(input_model, input_name)
            if node is None:
                raise Error('Cannot find location {} in the input model'.format(input_name))
            shape = None if isinstance(input_user_shapes, list) else input_user_shapes[input_name]
            if input_name in input_user_data_types and input_user_data_types[input_name] is not None:
                data_type = input_user_data_types[input_name]
                _input_shapes.append({'node': node, 'shape': shape, 'data_type': data_type})
            else:
                _input_shapes.append({'node': node, 'shape': shape})
    elif isinstance(input_user_shapes, np.ndarray):
        model_inputs = input_model.get_inputs()
        assert len(model_inputs) == 1
        _input_shapes.append({'node': model_inputs[0], 'shape': input_user_shapes})
    else:
        assert input_user_shapes is None
    # TODO: add logic for freeze_placeholder
    return _input_shapes, dict()


def fe_output_user_data_repack(input_model: InputModel, outputs: list):
    """

    :param input_model: Input Model to operate on
    :param outputs: list of node names provided by user
    :return: dictionary with node IDs as keys and list of port dictionaries as values
    Example of outputs dictionary:
    _outputs =
    {
        'node_ID':
            [
                {'out': 0},
                {'out': 1},
            ],
        'node_1_ID':
            [
                {'port': None}
            ],
        'node_2_ID':
            [
                {'in': 3}
            ]
    }
    """
    _outputs = []
    if outputs is not None and len(outputs) > 0:
        for output in outputs:
            node = decode_name_with_port(input_model, output)
            if node is None:
                raise Error('Cannot find location {} in the graph'.format(output))
            _outputs.append({'node': node})
    return _outputs


def fe_user_data_repack(input_model: InputModel, input_user_shapes: [None, list, dict, np.array],
                        input_user_data_types: dict, outputs: list, freeze_placeholder: dict):
    """
    :param input_model: Input Model to operate on
    :param input_user_shapes: data structure representing user input cutting request
    :param outputs: list of node names to treat as outputs
    :param freeze_placeholder: dictionary with placeholder names as keys and freezing value as values
    :return: restructured input, output and freeze placeholder dictionaries or None values
    """
    _input_shapes, _freeze_placeholder = fe_input_user_data_repack(
        input_model, input_user_shapes, freeze_placeholder, input_user_data_types=input_user_data_types)
    _outputs = fe_output_user_data_repack(input_model, outputs)

    print('---------- Inputs/outputs/freezePlaceholder -----------')
    print(_input_shapes)
    print(_outputs)
    print(freeze_placeholder)
    print('------------------------------------')

    return _input_shapes, _outputs, _freeze_placeholder
