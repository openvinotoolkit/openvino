# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import re
from enum import Enum

import numpy as np
from openvino._pyopenvino import Place, PartialShape

from openvino.frontend import InputModel  # pylint: disable=no-name-in-module,import-error
from openvino.tools.mo_lite.utils.clean_utils import raise_no_node, raise_node_name_collision
from openvino.tools.mo_lite.utils.error import Error


class IOType(Enum):
    Input = 1
    Output = 2


def decode_name_with_port(
        input_model: InputModel, node_name: str, framework="", io_type=IOType.Input
) -> Place or None:
    """
    Decode name with optional port specification w/o traversing all the nodes in the graph
    TODO: in future node_name can specify input/output port groups as well as indices (58562)
    :param input_model: Input Model
    :param node_name: user provided node name
    :return: decoded place in the graph
    """
    found_places = []
    found_place_names = []

    def get_place_by_operation_name(input_model, name, framework, io_type):
        node = input_model.get_place_by_operation_name(name)
        if node and framework == "onnx":
            if io_type == IOType.Input:
                return (
                    node.get_input_port(input_port_index=0)
                    .get_producing_port()
                    .get_target_tensor()
                )
            else:
                return node.get_output_port(output_port_index=0).get_target_tensor()
        return node

    # find by tensor name
    place = input_model.get_place_by_tensor_name(node_name)
    if place:
        found_place_names.append("Tensor:" + node_name)
        found_places.append(place)
    else:
        # find by operation name
        place = get_place_by_operation_name(input_model, node_name, framework, io_type)
        name = node_name
        if framework == "onnx" and io_type == IOType.Output:
            name = "Tensor:" + name

        if place:
            found_place_names.append(name)
            found_places.append(place)

    def try_get_node(model, name, framework):
        node = model.get_place_by_operation_name(name)
        if node:
            return node
        if framework == "onnx":
            tensor = model.get_place_by_tensor_name(name)
            if tensor:
                if tensor.is_input() or tensor.is_output():
                    return tensor
                return tensor.get_producing_operation()
        return None

    def get_port(match, match_starts_with_name, input_model, framework):
        if not match:
            return None

        if match_starts_with_name:
            name = match.group(1)
            port_index = match.group(2)
        else:
            name = match.group(2)
            port_index = match.group(1)

        node = try_get_node(input_model, name, framework)
        if node:
            # if regular expression has structure <name>:<port>, get node output port.
            # Otherwise get node input port
            if match_starts_with_name:
                return node.get_output_port(output_port_index=int(port_index))
            else:
                return node.get_input_port(input_port_index=int(port_index))
        else:
            None

    regexp_post = r"(.+):(\d+)"
    match = re.search(regexp_post, node_name)
    match_port = get_port(
        match=match,
        match_starts_with_name=True,
        input_model=input_model,
        framework=framework,
    )

    if match_port:
        name = match.group(1)
        if framework == "onnx":
            found_place_names.append("Tensor:" + name)
            found_places.append(match_port.get_target_tensor())
        else:
            found_place_names.append(name)
            found_places.append(match_port)

    regexp_pre = r"(\d+):(.+)"
    match = re.search(regexp_pre, node_name)
    match_port = get_port(
        match=match,
        match_starts_with_name=False,
        input_model=input_model,
        framework=framework,
    )

    if match_port:
        name = match.group(2)
        if framework == "onnx":
            found_place_names.append("Tensor:" + name)
            found_places.append(match_port.get_producing_port().get_target_tensor())
        else:
            found_places.append(match_port)
            found_place_names.append(name)

    if len(found_places) == 0:
        raise_no_node(node_name)

    # Check that there is no collision, all found places shall point to same data
    if not all([n.is_equal_data(found_places[0]) for n in found_places]):
        raise_node_name_collision(node_name, found_place_names)

    # TODO: Add support for input/output group name and port index here (58562)
    # For new frontends logic shall be extended to additionally support input and output group names
    return found_places[0]


def fe_input_user_data_repack(
        input_model: InputModel,
        input_user_shapes: [None, list, dict, np.ndarray],
        freeze_placeholder: dict,
        framework: str,
        input_user_data_types=None,
):
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
    _input_names = []
    if isinstance(input_user_shapes, list) and len(input_user_shapes) > 1 and isinstance(input_user_shapes[0],
                                                                                         PartialShape):
        for shape in input_user_shapes:
            assert isinstance(shape, PartialShape), "Got incorrect format of input shapes."
        model_inputs = input_model.get_inputs()
        assert len(model_inputs) == len(input_user_shapes)
        for idx, model_input in enumerate(model_inputs):
            _input_shapes.append({"node": model_input, "shape": input_user_shapes[idx]})
    elif isinstance(input_user_shapes, list) or isinstance(input_user_shapes, dict):
        for input_name in input_user_shapes:
            node = decode_name_with_port(
                input_model, input_name, framework, IOType.Input
            )
            if node is None:
                raise Error(
                    "Cannot find location {} in the input model".format(input_name)
                )
            shape = (
                None
                if isinstance(input_user_shapes, list)
                else input_user_shapes[input_name]
            )
            if isinstance(input_user_data_types, dict) and input_user_data_types.get(input_name) is not None:
                data_type = input_user_data_types[input_name]
                _input_shapes.append(
                    {
                        "node": node,
                        "shape": shape,
                        "data_type": data_type,
                        "input_name": input_name,
                    }
                )
            else:
                _input_shapes.append(
                    {
                        "node": node,
                        "shape": shape,
                        "input_name": input_name
                    }
                )
            _input_names.append(input_name)
    elif isinstance(input_user_shapes, PartialShape):
        # this branch covers the single use of `input_shape` without `input` option
        # but it can be used along with `freeze_placeholder_with_value` option
        # for example, --input_shape [3] --freeze_placeholder_with_value "is_training->False"
        # means the model has two inputs: one is is_training to be frozen, the other to re-write the shape
        # NOTE: the logic relies on parameters with the single name
        model_inputs = input_model.get_inputs()
        frozen_names = freeze_placeholder.keys()
        assert len(model_inputs) == len(frozen_names) + 1, "Please check the conversion command-line. " \
                                                           "Total number of model inputs must match to a number " \
                                                           "of input shapes along with frozen inputs."
        for node in model_inputs:
            assert len(node.get_names()) > 0, "Original input models must have names."
            input_name = node.get_names()[0]
            if input_name not in frozen_names:
                _input_shapes.append(
                    {
                        "node": node,
                        "shape": input_user_shapes,
                        "input_name": input_name
                    }
                )
                _input_names.append(input_name)
                break
    else:
        # this case means that we use original inputs of the model
        # and they should not be changed and their properties (shape and type) should not be over-written
        # NOTE: the logic relies on parameters with the single name
        assert input_user_shapes is None
        for node in input_model.get_inputs():
            assert len(node.get_names()) > 0, "Original input models must have names."
            input_name = node.get_names()[0]
            _input_shapes.append(
                {
                    "node": node,
                    "input_name": input_name
                }
            )
            # mark-up Place names we already put into the _input_names
            # to avoid duplicates in updates by freeze_placeholder below
            _input_names.append(input_name)

    if freeze_placeholder:
        # in case freezing via freeze_placeholder_with_value option, _input_shapes can miss some frozen places
        for input_name in freeze_placeholder:
            if input_name in _input_names:
                continue
            node = decode_name_with_port(
                input_model, input_name, framework, IOType.Input
            )
            _input_shapes.append(
                {
                    "node": node,
                    "input_name": input_name
                }
            )
        return _input_shapes, freeze_placeholder
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
            node = decode_name_with_port(input_model, output, framework, IOType.Output)
            if node is None:
                raise Error("Cannot find location {} in the graph".format(output))
            _outputs.append({"node": node})
    return _outputs


def fe_user_data_repack(
        input_model: InputModel,
        input_user_shapes: [None, list, dict, np.array],
        input_user_data_types: dict,
        outputs: list,
        freeze_placeholder: dict,
        framework: str,
):
    """
    :param input_model: Input Model to operate on
    :param input_user_shapes: data structure representing user input cutting request
    :param input_user_data_types: dictionary with input nodes and its data types
    :param outputs: list of node names to treat as outputs
    :param freeze_placeholder: dictionary with placeholder names as keys and freezing value as values
    :return: restructured input, output and freeze placeholder dictionaries or None values
    """
    _input_shapes, _freeze_placeholder = fe_input_user_data_repack(
        input_model,
        input_user_shapes,
        freeze_placeholder,
        framework,
        input_user_data_types=input_user_data_types,
    )
    _outputs = fe_output_user_data_repack(input_model, outputs, framework)

    return _input_shapes, _outputs, _freeze_placeholder
