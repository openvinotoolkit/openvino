# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
import re
from collections import defaultdict
from copy import copy

import numpy as np

from mo.utils.error import Error

from ngraph.frontend import InputModel  # pylint: disable=no-name-in-module,import-error


def decode_name_with_port(input_model: InputModel, node_name: str):
    """
    Decode name with optional port specification w/o traversing all the nodes in the graph
    TODO: in future node_name can specify input/output port groups and indices (58562)
    :param input_model: Input Model
    :param node_name: user provided node name
    :return: decoded place in the graph
    """
    # Check exact match with one of the names in the graph first
    node = input_model.get_place_by_tensor_name(node_name)
    if node:
        return node

    # TODO: Add support for input/output group name and port index here (58562)
    # Legacy frontends use format "number:name:number" to specify input and output port indices
    # For new frontends this logic shall be extended to additionally support input and output group names
    raise Error('There is no node with name {}'.format(node_name))


def fe_input_user_data_repack(input_model: InputModel, input_user_shapes: [None, list, dict, np.ndarray],
                              freeze_placeholder: dict, input_user_data_types=dict()):
    """
    Restructures user input cutting request. Splits ports out of node names.
        Transforms node names to node ids.
    :param input_model: current input model
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
            if input_user_data_types.get(input_name) is not None:
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
    # TODO: implement freeze_placeholder (issue 58560)
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
    if outputs is not None:
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
    :param input_user_data_types: dictionary with input nodes and its data types
    :param outputs: list of node names to treat as outputs
    :param freeze_placeholder: dictionary with placeholder names as keys and freezing value as values
    :return: restructured input, output and freeze placeholder dictionaries or None values
    """
    _input_shapes, _freeze_placeholder = fe_input_user_data_repack(
        input_model, input_user_shapes, freeze_placeholder, input_user_data_types=input_user_data_types)
    _outputs = fe_output_user_data_repack(input_model, outputs)

    return _input_shapes, _outputs, _freeze_placeholder
