# type: ignore
from enum import Enum
from __future__ import annotations
from openvino._pyopenvino import InputModel
from openvino._pyopenvino import PartialShape
from openvino._pyopenvino import Place
from openvino.tools.ovc.error import Error
import enum
import numpy
import numpy as np
import openvino._pyopenvino
import re as re
import typing
__all__ = ['Enum', 'Error', 'IOType', 'InputModel', 'PartialShape', 'Place', 'convert_params_lists_to_dicts', 'decode_name_with_port', 'fe_input_user_data_repack', 'fe_output_user_data_repack', 'fe_user_data_repack', 'find_first_unused_input', 'np', 'raise_no_node', 'raise_node_name_collision', 're']
class IOType(enum.Enum):
    Input: typing.ClassVar[IOType]  # value = <IOType.Input: 1>
    Output: typing.ClassVar[IOType]  # value = <IOType.Output: 2>
def convert_params_lists_to_dicts(input_model, input_user_shapes: [list, dict], input_user_data_types: [list, dict]):
    """
    
        Convert lists of unnamed params to dicts using input names from input_model.
    
        :param input_model: openvino.InputModel
        :param input_user_shapes: list of input shapes or dictionary where key is input name, value is input shape from user.
        :param input_user_data_types: list of input types or dictionary where key is input name, value is input type from user.
    
        :return: (input_user_shapes_dict, input_user_data_types_dict, freeze_placeholder), where
        input_user_shapes_dict - dictionary where key is input name, value is shape from user;
        input_user_data_types_dict - dictionary where key is input name, value is type from user;
        freeze_placeholder - dictionary where key is input name, value is input value from user;
        
    """
def decode_name_with_port(input_model: openvino._pyopenvino.InputModel, node_name: str, framework = '', io_type = ...) -> openvino._pyopenvino.Place:
    """
    
        Decode name with optional port specification w/o traversing all the nodes in the graph
        TODO: in future node_name can specify input/output port groups as well as indices (58562)
        :param input_model: Input Model
        :param node_name: user provided node name
        :return: decoded place in the graph
        
    """
def fe_input_user_data_repack(input_model: openvino._pyopenvino.InputModel, input_user_shapes: [None, list, dict, numpy.ndarray], freeze_placeholder: dict, framework: str, input_user_data_types = None):
    """
    
        Restructures user input cutting request. Splits ports out of node names.
            Transforms node names to node ids.
        :param input_model: current input model
        :param input_user_shapes: data structure representing user input cutting request. It may be:
        # None value if user did not provide neither "input" nor "input_shape" keys
        # list instance which contains input layer names with or without ports if user provided
            only "input" key
        # dict instance which contains input layer names with or without ports as keys and shapes as
            values if user provided both "input" and "input_shape"
        # np.ndarray if user provided only "input_shape" key
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
def fe_output_user_data_repack(input_model: openvino._pyopenvino.InputModel, outputs: list, framework: str):
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
def fe_user_data_repack(input_model: openvino._pyopenvino.InputModel, input_user_shapes: [None, list, dict, numpy.array], input_user_data_types: dict, outputs: list, freeze_placeholder: dict, framework: str):
    """
    
        :param input_model: Input Model to operate on
        :param input_user_shapes: data structure representing user input cutting request
        :param input_user_data_types: dictionary with input nodes and its data types
        :param outputs: list of node names to treat as outputs
        :param freeze_placeholder: dictionary with placeholder names as keys and freezing value as values
        :return: restructured input, output and freeze placeholder dictionaries or None values
        
    """
def find_first_unused_input(model_inputs: list, param_dict: dict, param_name: str):
    """
    
        Finds first input in model_inputs, which is not present in freeze_placeholder dictionary or param_dict.
    
        :param model_inputs: list of model inputs
        :param param_dict: dictionary where key is input name, value is parameter value (shape or type).
        :param param_name: name of parameter used in exception message.
    
        :return: first input name, which is not present in freeze_placeholder dictionary or param_dict.
        
    """
def raise_no_node(node_name: str):
    ...
def raise_node_name_collision(node_name: str, found_nodes: list):
    ...
