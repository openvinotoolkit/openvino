# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import re

from openvino.tools.mo.front.extractor import raise_no_node, raise_node_name_collision
from openvino.tools.mo.utils.error import Error

from openvino.frontend import InputModel  # pylint: disable=no-name-in-module,import-error

import numpy as np


def decode_name_with_port(input_model: InputModel, node_name: str, framework="", cut_type=""):
    """
    Decode name with optional port specification w/o traversing all the nodes in the graph
    TODO: in future node_name can specify input/output port groups as well as indices (58562)
    :param input_model: Input Model
    :param node_name: user provided node name
    :return: decoded place in the graph
    """
    found_nodes = []
    found_node_names = []

    def extract_node(node_name: str, cut_type: str):
        tensor = input_model.get_place_by_tensor_name(node_name)
        if tensor:
            return tensor
        if cut_type == "output":
            if node_name.count(":"):
                split = node_name.split(":")
                # assuming port there are at max 10 ports
                if split[0].isnumeric() and len(split[0]) == 1:
                    node = input_model.get_place_by_operation_name(split[1])
                    if node:
                        return node.get_source_tensor(input_port_index=int(split[0]))
                    return input_model.get_place_by_tensor_name(split[1])
                node = input_model.get_place_by_operation_name(split[0])
                if node:
                    return node.get_target_tensor(output_port_index=int(split[1]))
                return input_model.get_place_by_tensor_name(split[0])
            node = input_model.get_place_by_operation_name(node_name)
            return node.get_target_tensor(output_port_index=0)
        if cut_type == "input":
            if node_name.count(":"):
                split = node_name.split(":")
                # assuming port there are at max 10 ports
                if split[0].isnumeric() and len(split[0]) == 1:
                    node = input_model.get_place_by_operation_name(split[1])
                    if node:
                        return node.get_source_tensor(input_port_index=int(split[0]))
                    return input_model.get_place_by_tensor_name(split[1])
                node = input_model.get_place_by_operation_name(split[0])
                if node:
                    post_node = node.get_consuming_operations()[0]
                    return post_node.get_source_tensor(input_port_index=int(split[1]))
                return input_model.get_place_by_tensor_name(split[0])
            node = input_model.get_place_by_operation_name(node_name)
            return node.get_source_tensor(input_port_index=0)
        return None

    node = extract_node(node_name, cut_type=cut_type)

    if node:
        found_node_names.append('Tensor:' + node_name)
        found_nodes.append(node)

    if len(found_nodes) == 0:
        raise_no_node(node_name)

    # Check that there is no collision, all found places shall point to same data
    if not all([n.is_equal_data(found_nodes[0]) for n in found_nodes]):
        raise_node_name_collision(node_name, found_node_names)

    # TODO: Add support for input/output group name and port index here (58562)
    # For new frontends logic shall be extended to additionally support input and output group names
    return found_nodes[0]


def fe_input_user_data_repack(input_model: InputModel, input_user_shapes: [None, list, dict, np.ndarray],
                              freeze_placeholder: dict, framework: str, input_user_data_types=dict(), ):
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
            node = decode_name_with_port(input_model, input_name, framework, "input")
            if node is None:
                raise Error('Cannot find location {} in the input model'.format(input_name))
            shape = None if isinstance(input_user_shapes, list) else input_user_shapes[input_name]
            if input_user_data_types.get(input_name) is not None:
                data_type = input_user_data_types[input_name]
                _input_shapes.append({'node': node, 'shape': shape, 'data_type': data_type, 'input_name': input_name})
            else:
                _input_shapes.append({'node': node, 'shape': shape, 'input_name': input_name})
    elif isinstance(input_user_shapes, tuple):
        model_inputs = input_model.get_inputs()
        assert len(model_inputs) == 1
        _input_shapes.append({'node': model_inputs[0], 'shape': input_user_shapes})
    else:
        assert input_user_shapes is None
    # TODO: implement freeze_placeholder (issue 58560)
    return _input_shapes, dict()


def fe_output_user_data_repack(input_model: InputModel, outputs: list, framework: str):
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
            node = decode_name_with_port(input_model, output, framework, "output")
            if node is None:
                raise Error('Cannot find location {} in the graph'.format(output))
            _outputs.append({'node': node})
    return _outputs


def fe_user_data_repack(input_model: InputModel, input_user_shapes: [None, list, dict, np.array],
                        input_user_data_types: dict, outputs: list, freeze_placeholder: dict, framework: str):
    """
    :param input_model: Input Model to operate on
    :param input_user_shapes: data structure representing user input cutting request
    :param input_user_data_types: dictionary with input nodes and its data types
    :param outputs: list of node names to treat as outputs
    :param freeze_placeholder: dictionary with placeholder names as keys and freezing value as values
    :return: restructured input, output and freeze placeholder dictionaries or None values
    """
    _input_shapes, _freeze_placeholder = fe_input_user_data_repack(
        input_model, input_user_shapes, freeze_placeholder,  framework, input_user_data_types=input_user_data_types,)
    _outputs = fe_output_user_data_repack(input_model, outputs, framework)
    return _input_shapes, _outputs, _freeze_placeholder
