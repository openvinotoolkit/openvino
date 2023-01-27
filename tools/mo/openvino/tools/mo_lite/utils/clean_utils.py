# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
import numpy as np
import logging as log
from openvino.tools.mo_lite.utils.error import Error


def default_path():
    EXT_DIR_NAME = '.'
    return os.path.abspath(os.getcwd().join(EXT_DIR_NAME))

dynamic_dimension = np.ma.masked

def validate_batch_in_shape(shape, layer_name: str):
    """
    Raises Error #39 if shape is not valid for setting batch size
    Parameters
    ----------
    shape: current shape of layer under validation
    layer_name: name of layer under validation
    """
    from openvino.tools.mo_lite.utils.utils import refer_to_faq_msg
    if len(shape) == 0 or (shape[0] is not dynamic_dimension and shape[0] not in (-1, 0, 1)):
        raise Error(('The input layer {} has a shape {} defined in the model. \n\n' +
                     'When you use -b (--batch) option, Model Optimizer applies its value to the first ' +
                     'element of the shape if it is equal to -1, 0 or 1. Otherwise, this is the ambiguous ' +
                     'situation - Model Optimizer can not know in advance whether the layer has the batch ' +
                     'dimension or not.\n\n For example, you want to set batch dimension equals 100 ' +
                     'for the input layer "data" with shape (10,34). Although you can not use --batch, ' +
                     'you should pass --input_shape (100,34) instead of --batch 100. \n\n' +
                     'You can also tell Model Optimizer where batch dimension is located by specifying --layout. \n\n' +
                     refer_to_faq_msg(39))
                    .format(layer_name, shape))


def raise_no_node(node_name: str):
    raise Error('No node with name {}'.format(node_name))


def raise_node_name_collision(node_name: str, found_nodes: list):
    raise Error('Name collision was found, there are several nodes for mask "{}": {}. '
                'If your intention was to specify port for node, please instead specify node names connected to '
                'this port. If your intention was to specify the node name, please add port to the node '
                'name'.format(node_name, found_nodes))


def get_enabled_and_disabled_transforms():
    """
    :return: tuple of lists with force enabled and disabled id of transformations.
    """
    disabled_transforms = os.environ['MO_DISABLED_TRANSFORMS'] if 'MO_DISABLED_TRANSFORMS' in os.environ else ''
    enabled_transforms = os.environ['MO_ENABLED_TRANSFORMS'] if 'MO_ENABLED_TRANSFORMS' in os.environ else ''

    assert isinstance(enabled_transforms, str)
    assert isinstance(disabled_transforms, str)

    disabled_transforms = disabled_transforms.split(',')
    enabled_transforms = enabled_transforms.split(',')

    return enabled_transforms, disabled_transforms


def split_node_in_port(node_id: str):
    """Split node_id in form port:node to separate node and port, where port is converted to int"""
    if isinstance(node_id, str):
        separator = ':'
        parts = node_id.split(separator)
        if len(parts) > 1:
            if parts[0].isdigit():
                node_name = separator.join(parts[1:])
                try:
                    port = int(parts[0])
                    return node_name, port
                except ValueError as err:
                    log.warning('Didn\'t recognize port:node format for "{}" because port is not an integer.'.format(
                    node_id))
            else:
                node_name = separator.join(parts[:-1])
                try:
                    port = int(parts[-1])
                    return node_name, port
                except ValueError as err:
                    log.warning('Didn\'t recognize node:port format for "{}" because port is not an integer.'.format(
                    node_id))

    return node_id, None


"""
Packed data of custom types are stored in numpy uint8 data type.
To distinguish true uint8 and custom data we introduce this class not to store,
but to have unique data type in SUPPORTED_DATA_TYPES map
"""


class packed_U1(np.generic):
       pass


class packed_U4(np.generic):
    pass


class packed_I4(np.generic):
    pass


SUPPORTED_DATA_TYPES = {
    'float': (np.float32, 'FP32', 'f32'),
    'half': (np.float16, 'FP16', 'f16'),
    'FP32': (np.float32, 'FP32', 'f32'),
    'FP64': (np.float64, 'FP64', 'f64'),
    'FP16': (np.float16, 'FP16', 'f16'),
    'I32': (np.int32, 'I32', 'i32'),
    'I64': (np.int64, 'I64', 'i64'),
    'int8': (np.int8, 'I8', 'i8'),
    'int32': (np.int32, 'I32', 'i32'),
    'int64': (np.int64, 'I64', 'i64'),
    'bool': (bool, 'BOOL', 'boolean'),
    'uint8': (np.uint8, 'U8', 'u8'),
    'uint32': (np.uint32, 'U32', 'u32'),
    'uint64': (np.uint64, 'U64', 'u64'),

    # custom types
    'U1': (packed_U1, 'U1', 'u1'),
    'int4': (packed_I4, 'I4', 'i4'),
    'uint4': (packed_U4, 'U4', 'u4'),
    'I4': (packed_I4, 'I4', 'i4'),
    'U4': (packed_U4, 'U4', 'u4'),
}


def destination_type_to_np_data_type(dst_type):
    for np_t, _, destination_type in SUPPORTED_DATA_TYPES.values():
        if destination_type == dst_type:
            return np_t
    raise Error('Destination type "{}" is not supported'.format(dst_type))


def np_data_type_to_destination_type(np_data_type):
    for np_t, _, destination_type in SUPPORTED_DATA_TYPES.values():
        if np_t == np_data_type:
            return destination_type
    raise Error('Data type "{}" is not supported'.format(np_data_type))


def get_ir_version(argv: argparse.Namespace):
    """
    Determine IR version based on command line arguments and the default version.
    :param argv: the parsed command line arguments
    :return: the IR version
    """
    return 11


def get_new_placeholder_name(node_id: str, is_out_port: bool = False, port: int = 0):
    """
    Forms a name of new placeholder created by cutting a graph
    :param node_id: a node name that is cut
    :param is_out_port: it is True iff output port is cut
    :param port: a port number
    :return: a name of new placeholder created by cutting a graph
    """
    port_type = '_out' if is_out_port else ''
    return '{}/placeholder{}_port_{}'.format(node_id, port_type, port)


def create_params_with_custom_types(packed_user_shapes: [None, dict]):
    """
    Compute a list of placeholder names for which an user specifies custom type
    :param packed_user_shapes: packed data that contains input node names,
    their port numbers, shapes and data types
    :return: a list of placeholder names for which an user specifies custom type
    Example of packed_user_shapes dictionary:
    packed_user_shapes =
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
    For which the function returns a list ['node_1_ID'] because this node only has custom data type
    """
    if packed_user_shapes is None:
        return []

    params_with_custom_types = []
    for input_name in packed_user_shapes:
        for desc in packed_user_shapes[input_name]:
            p_name = input_name
            if 'port' in desc and desc['port'] is None:  # neither input nor output port specified
                user_defined_type = desc.get('data_type', None)
            else:  # need to check the particular port the Parameter was created for
                p_name = get_new_placeholder_name(input_name, 'out' in desc,
                                                  desc['out'] if 'out' in desc else desc['in'])
                user_defined_type = desc.get('data_type', None)
            if user_defined_type is not None:
                params_with_custom_types.append(p_name)
    return params_with_custom_types
