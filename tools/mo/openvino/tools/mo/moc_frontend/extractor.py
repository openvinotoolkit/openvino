# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import re

from openvino.tools.mo.front.extractor import raise_no_node, raise_node_name_collision
from openvino.tools.mo.utils.error import Error

from openvino.frontend import InputModel  # pylint: disable=no-name-in-module,import-error

import numpy as np

from enum import Enum

class MatchType(Enum):
    PRE = 1
    POST = 2


def decode_name_with_port(input_model: InputModel, node_name: str, framework=""):
    """
    Decode name with optional port specification w/o traversing all the nodes in the graph
    TODO: in future node_name can specify input/output port groups as well as indices (58562)
    :param input_model: Input Model
    :param node_name: user provided node name
    :return: decoded place in the graph
    """

    def extract_nodes(input_model, name, port, match_type, search_tensor = True):
        nodes = []
        node_names = []
        node = input_model.get_place_by_operation_name(name)
        if not node and search_tensor:
            tensor = input_model.get_place_by_tensor_name(name)
            if tensor:
                node_names.append('Tensor:' + tensor.get_names()[0])
                nodes.append(tensor)
                search_tensor = False
        if node:
            # if there is an operation with given name, we add input port
            if match_type == MatchType.PRE:
                new_node = node.get_input_port(input_port_index=int(port))
            elif match_type == MatchType.POST:
                new_node = node.get_output_port(output_port_index=int(port))
            if new_node:
                node_names.append(name)
                nodes.append(new_node)
            # if we are still looking for the tensor e add one with given port
            if search_tensor:
                if match_type == MatchType.PRE:
                    tensor = node.get_source_tensor(input_port_index=int(port))
                elif match_type == MatchType.POST:
                    tensor = node.get_target_tensor(output_port_index=0)
                if tensor:
                    node_names.append('Tensor:' + tensor.get_names()[0])
                    nodes.append(tensor)
                    search_tensor = False
        return nodes, node_names, search_tensor


    def try_get_nodes(input_model, node_name):
        # Passed node_name can be in several forms:
        # (1) name (2) port:name (3) name:port
        found_nodes = []
        found_node_names = []
        # if we find tensor, there is no need to continue searching
        search_tensor = True
        # check if there is a tensor with given node_name
        tensor = input_model.get_place_by_tensor_name(node_name)
        if tensor:
            found_node_names.append('Tensor:' + tensor.get_names()[0])
            found_nodes.append(tensor)
            search_tensor = False

        regexp_pre = r'(\d+):(.+)'
        match_pre = re.search(regexp_pre, node_name)
        # we check for port:name combination
        if match_pre:
            nodes, node_names, search_tensor = extract_nodes(input_model, match_pre.group(2), match_pre.group(1), MatchType.PRE, search_tensor)
            if nodes:
                found_nodes += nodes
                found_node_names += node_names
        regexp_post = r'(.+):(\d+)'
        match_post = re.search(regexp_post, node_name)
        # we check for name:port combination
        if match_post:
            nodes, node_names, search_tensor = extract_nodes(input_model, match_post.group(1), match_post.group(2), MatchType.POST, search_tensor)
            if nodes:
                found_nodes += nodes
                found_node_names += node_names
        # if node and tensor were not found yet
        # we try to find operation with node_name
        if not found_nodes and search_tensor:
            node = input_model.get_place_by_operation_name(node_name)
            if node:
                tensor = node.get_target_tensor(output_port_index=0)
                if tensor:
                    found_node_names.append('Tensor:' + tensor.get_names()[0])
                    found_nodes.append(tensor)
        return found_node_names, found_nodes

    found_node_names, found_nodes = try_get_nodes(input_model, node_name)
    if len(found_nodes) == 0:
        raise_no_node(node_name)

    # Check that there is no collision, all found places shall point to same data
    if not all([n.is_equal_data(found_nodes[0]) for n in found_nodes]):
        raise_node_name_collision(node_name, found_node_names)

    # TODO: Add support for input/output group name and port index here (58562)
    # For new frontends logic shall be extended to additionally support input and output group names
    idx = next((idx for idx, name in enumerate(found_node_names) if 'Tensor' in name), 0)
    return found_nodes[idx]


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
            node = decode_name_with_port(input_model, input_name, framework)
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
            node = decode_name_with_port(input_model, output, framework)
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
